"""One licensed 720p road excerpt: aligned CRF23 direct/bicubic/native NanoVSR."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
from streaming_common import ROOT,dump,exact
sys.path.insert(0,str(ROOT/'client'))
from streaming_client import load_model
from evaluate_x4_video import metrics,metric_checks
OUT=ROOT/'results/native720-road-2026-10-18'
EXPORT=ROOT/'presentation_assets/native720'
N=300
commands=[]
def run(cmd):
    cmd=list(map(str,cmd));commands.append(cmd);dump(OUT/'commands.json',commands)
    return subprocess.check_output(cmd)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csvwrite(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def decode(path,w,h,target):
    cmd=['ffmpeg','-v','error','-i',str(path),'-an','-fps_mode','passthrough','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    commands.append(cmd);dump(OUT/'commands.json',commands)
    a=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(N,h,w,3))
    with subprocess.Popen(cmd,stdout=subprocess.PIPE) as p:
        for i in range(N):a[i]=np.frombuffer(exact(p.stdout,w*h*3),np.uint8).reshape(h,w,3)
        assert not p.stdout.read(1) and p.wait()==0
    a.flush();return a

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    assert not (EXPORT/'quality.json').exists(), 'Completed comparison already exists'
    clip=next(c for c in json.loads((ROOT/'configs/natural-study.json').read_text())['clips'] if c['id']=='road')
    source=ROOT/clip['source']
    old=next(c for c in json.loads((ROOT/'results/natural-2026-10-15/provenance.json').read_text()) if c['id']=='road')
    assert sha(source)==old['sha256']
    probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',source]));v=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (v['width'],v['height'])==(1280,720)
    ref=OUT/'reference.nut'
    if not ref.exists():run(['ffmpeg','-v','error','-ss','10','-i',source,'-an','-vf','setpts=PTS-STARTPTS,fps=30,format=bgr0','-frames:v',N,'-c:v','ffv1','-level','3','-threads','1',ref])
    a=decode(ref,1280,720,OUT/'reference.npy')
    csvwrite(OUT/'alignment.csv',[dict(frame=i,reference_pts=i/30,rgb_sha256=hashlib.sha256(a[i].tobytes()).hexdigest()) for i in range(N)])
    panels=[cv2.resize(a[i],(640,360)) for i in (0,150,299)]
    cv2.imwrite(str(EXPORT/'source-contact.png'),np.concatenate(panels,axis=1)[...,::-1])
    dump(EXPORT/'provenance.json',dict(**clip,sha256=sha(source),source_resolution=[1280,720],reference_resolution=[1280,720],reference_sha256=sha(ref),reference_fps=30,conversion='Source seconds10-20; fps50 to30 by timestamp-based nearest duplication/drop; no spatial scaling; lossless RGB FFV1/NUT. Already compressed source, not pristine ground truth.'))
    for width in (320,1280):
        path=OUT/f'w{width}-crf23.ts'
        run(['ffmpeg','-v','error','-i',ref,'-an','-frames:v',N,'-vf',f'scale={width}:{width*9//16}:flags=bicubic:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-preset','veryfast','-tune','zerolatency','-crf','23','-threads','1','-g','30','-keyint_min','30','-sc_threshold','0','-bf','0','-refs','1','-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-mpegts_copyts','1','-muxdelay','0','-muxpreload','0','-f','mpegts',path])
        packets=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-of','json',path]))['packets']
        pts=[float(p['pts_time']) for p in packets];pos=[int(p['pos']) for p in packets]
        assert len(pts)==N and pos==sorted(set(pos)) and all(p%188==0 for p in pos)
        np.testing.assert_allclose(np.array(pts)-pts[0],np.arange(N)/30,atol=2e-6,rtol=0)
        dump(path.with_suffix('.json'),dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,width=width,height=width*9//16,frames=N,pts=pts,offsets=[0]+pos[1:]+[path.stat().st_size],crf=23))
        decode(path,width,width*9//16,OUT/f'w{width}-decoded.npy')
    print('Prepared native720 reference and shared LR asset.',flush=True)

def score():
    assert not (EXPORT/'quality.json').exists(),'Refusing to overwrite completed comparison'
    torch,model=load_model(dict(checkpoint='external/nanovsr/checkpoints/nanovsr_226k.pth'))
    hr=np.load(OUT/'reference.npy',mmap_mode='r');lr=np.load(OUT/'w320-decoded.npy',mmap_mode='r');direct=np.load(OUT/'w1280-decoded.npy',mmap_mode='r')
    arrays={m:np.lib.format.open_memmap(OUT/f'w320-crf23-{m}.npy',mode='w+',dtype=np.uint8,shape=(N,720,1280,3)) for m in ('bicubic','nanovsr')}
    for start in range(0,N,3):
        x=torch.from_numpy(lr[start:start+3].copy()).permute(0,3,1,2).unsqueeze(0).float().cuda()/255
        with torch.inference_mode():
            y=model(x);assert y.shape==(1,3,3,720,1280) and torch.isfinite(y).all()
            arrays['nanovsr'][start:start+3]=y[0].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
        arrays['bicubic'][start:start+3]=[cv2.resize(f,(1280,720),interpolation=cv2.INTER_CUBIC) for f in lr[start:start+3]]
        del x,y
    for a in arrays.values():a.flush()
    # Compatibility with the independent streaming verifier, without duplicating data.
    direct_path=OUT/'w1280-crf23-direct.npy'
    if not direct_path.exists():direct_path.hardlink_to(OUT/'w1280-decoded.npy')
    rows=[];summary=[]
    for mode,a in [('direct',direct),*arrays.items()]:
        scores=[]
        for i in range(N):
            psnr,ssim=metrics(a[i],hr[i]);r=dict(method=mode,frame=i,pts=i/30,psnr_db=psnr,ssim=ssim);rows.append(r);scores.append(r)
            if i%60==0:print(f'{mode}: scored {i}/{N}',flush=True)
        width=1280 if mode=='direct' else 320;meta=json.loads((OUT/f'w{width}-crf23.json').read_text())
        summary.append(dict(method=mode,crf=23,frames=N,width=width,playback_size=[1280,720],bytes=meta['bytes'],kbps=meta['bytes']*8/10/1000,psnr_db=float(np.mean([r['psnr_db'] for r in scores])),ssim=float(np.mean([r['ssim'] for r in scores])),asset_sha256=meta['sha256']))
    csvwrite(EXPORT/'frame-metrics.csv',rows);csvwrite(EXPORT/'comparison.csv',summary)
    dump(EXPORT/'quality.json',dict(playback_size=[1280,720],checkpoint_sha256=sha(ROOT/'external/nanovsr/checkpoints/nanovsr_226k.pth'),chunk_size=3,reference_sha256=sha(OUT/'reference.nut'),metric_protocol=metric_checks()+'; all frames RGB uint8, float64 PSNR and Gaussian11x11 SSIM; mean per-frame scores',comparison='Same CRF23, NOT matched bitrate; B/C identical encoded asset; NanoVSR native x4, no final spatial resize.',summary=summary))
    for mode in ('direct','bicubic','nanovsr'):
        c=json.loads((ROOT/'presentation_assets/configs/nanovsr-native.json').read_text())
        c.update(source=str((OUT/'reference.nut').relative_to(ROOT)),encoded_asset=str((OUT/f'w{1280 if mode=="direct" else 320}-crf23.ts').relative_to(ROOT)),native_output=mode=='nanovsr',warmup_chunks=3 if mode=='nanovsr' else 0,warmup_pipeline=mode=='nanovsr',quality_results='presentation_assets/native720/quality.json')
        dump(EXPORT/f'{mode}.json',c)
    panels=[]
    for label,a in [('Reference',hr),('Direct CRF23',direct),('Bicubic CRF23',arrays['bicubic']),('NanoVSR native x4',arrays['nanovsr'])]:
        crop=a[150,300:540,500:820].copy();crop=cv2.copyMakeBorder(crop,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20));cv2.putText(crop,label,(5,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);panels.append(crop)
    cv2.imwrite(str(EXPORT/'comparison-crops.png'),np.concatenate(panels,axis=1)[...,::-1])
    dump(EXPORT/'crop-selection.json',dict(frame=150,box=[500,300,320,240],scale='native 1:1',selection='fixed road/grass region; no metric-based search'))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score']);args=p.parse_args()
    if (OUT/'commands.json').exists():commands=json.loads((OUT/'commands.json').read_text())
    prepare() if args.stage=='prepare' else score()
