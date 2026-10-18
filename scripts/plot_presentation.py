"""Generate report figures using system Matplotlib; does not modify model environments."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/btp-matplotlib')
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'presentation_assets'
def read(name):return list(csv.DictReader(open(out/'data'/name)))
def save(name):
    plt.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');plt.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight');plt.close()
q=read('quality-all-points.csv');colors={'direct':'#2455a4','bicubic':'#d78317','nanovsr':'#25844c'}
for clip in ('flowers','road','waterfall'):
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for mode in colors:
        rows=sorted([r for r in q if r['clip']==clip and r['method']==mode],key=lambda r:float(r['kbps']))
        for ax,key,label in zip(axs,('psnr_db','ssim'),('RGB PSNR (dB)','RGB SSIM')):
            ax.plot([float(r['kbps']) for r in rows],[float(r[key]) for r in rows],'o-',label=mode,color=colors[mode])
            if mode=='direct':
                extra=[r for r in rows if float(r['crf']) not in (18,23,28)]
                ax.scatter([float(r['kbps']) for r in extra],[float(r[key]) for r in extra],marker='*',s=140,color=colors[mode],zorder=5)
            ax.set(xlabel='MPEG-TS payload rate (kbps)',ylabel=label);ax.grid(alpha=.25)
    axs[0].legend();fig.suptitle(f'{clip}: all aligned frames; stars = targeted direct encodes');fig.tight_layout();save(f'quality-rate-{clip}')
r=read('valid-repetitions.csv');fig,axs=plt.subplots(1,3,figsize=(14,4))
labels=[];groups=[]
for clip in ('flowers','road','waterfall'):
 for mode,warm in [('direct',0),('bicubic',0),('nanovsr',0),('nanovsr',3)]:
    labels.append(clip+'\n'+mode+('\nwarm' if warm else '\ncold'))
    groups.append([x for x in r if x['clip']==clip and x['mode']==mode and int(x['warmup_chunks'])==warm and float(x['buffer_ms'])==350])
for i,g in enumerate(groups):
 for ax,key in zip(axs,('latency_p95_ms','active_submission_fps','completeness')):
    vals=[float(x[key])*(100 if key=='completeness' else 1) for x in g]
    ax.scatter([i-.12,i,i+.12],vals,s=18,color=colors[g[0]['mode']])
for ax,title,yl in zip(axs,('Application latency','Active unique submissions','Playback completeness'),('Per-run p95 (ms)','Submissions / second','Submitted / source frames (%)')):
 ax.set(title=title,ylabel=yl,xticks=range(len(labels)),xticklabels=labels);ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.grid(axis='y',alpha=.25)
axs[0].axhline(350,color='gray',ls=':',label='Configured buffer');axs[0].legend(fontsize=8);axs[1].axhline(30,color='gray',ls=':');axs[2].set_ylim(95,101)
fig.suptitle('Three valid repetitions each, 350 ms buffer; dots are runs, not confidence intervals');fig.tight_layout();save('latency-playback')
if (out/'data/stability-windows.csv').exists():
 s=read('stability-windows.csv');fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True);x=[float(v['source_start_s']) for v in s]
 for key in ('latency_p50_ms','latency_p95_ms','queue_age_p95_ms'):axs[0].plot(x,[float(v[key]) if v[key] else np.nan for v in s],label=key)
 axs[0].set_ylabel('Milliseconds');axs[0].legend();axs[1].plot(x,[float(v['active_submission_fps']) for v in s]);axs[1].axhline(30,ls=':',color='gray');axs[1].set_ylabel('Active submissions/s')
 axs[2].bar(x,[float(v['drops']) for v in s],width=8);axs[2].set(ylabel='Drops / 300 source frames',xlabel='Source schedule window start (s)')
 for ax in axs:ax.grid(alpha=.2)
 fig.suptitle('Single looped-road stability run; 10 s source windows, not independent clips');fig.tight_layout();save('stability-trends')
fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
boxes=[(.02,'SERVER\nLive encode OR pre-encoded replay\nH.264 / MPEG-TS, paced PTS'),(.36,'LOCALHOST NETWORK\nFramed TCP records\nHeader → data → explicit EOS'),(.70,'CLIENT\nFFmpeg decode → bounded queue6\nDirect / bicubic / native-x4 NanoVSR\nResize → timed OpenCV playback')]
for x,t in boxes:ax.text(x,.6,t,transform=ax.transAxes,va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.8',fc='#e9f0f7',ec='#345'))
for x in (.295,.64):ax.annotate('',xy=(x+.05,.6),xytext=(x,.6),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2))
ax.text(.02,.08,'Natural study: replay excludes live encoding cost. Same-host monotonic schedule → GUI call return = application latency.\nNanoVSR chunk3: up to 66.7 ms lookahead; 320×180 → 1280×720 → 640×360; recurrent reset every chunk.',transform=ax.transAxes,fontsize=10)
save('architecture')
print('Saved PNG and SVG report figures.')
