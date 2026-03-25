import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ── Load SheetJS dynamically ──────────────────────────────────────────────────
function useXLSX() {
  const [ready, setReady] = useState(!!window.XLSX);
  useEffect(() => {
    if (window.XLSX) { setReady(true); return; }
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js";
    s.onload = () => setReady(true);
    document.head.appendChild(s);
  }, []);
  return ready;
}

// ── Color palette ─────────────────────────────────────────────────────────────
const C = {
  navy:"#0D1B2A", ink:"#1C2B3A", mid:"#274060", blue:"#2C6FAC",
  posShap:"#C0392B", negShap:"#1A5F96", bg:"#F2F5F9", paper:"#FFFFFF",
  gray:"#5D6D7E", light:"#E8EDF3", border:"#CDD5DF", accent:"#2980B9"
};
const TAB_COLS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f","#aec7e8","#ffbb78"];
const dlBtn = { marginTop:6, padding:"4px 14px", fontSize:12, background:C.navy, color:"#fff", border:"none", borderRadius:4, cursor:"pointer", fontFamily:"monospace", letterSpacing:0.3 };
const cardStyle = { background:C.paper, borderRadius:8, padding:"20px 24px", boxShadow:"0 1px 4px rgba(0,0,0,0.08)", marginBottom:16 };

// ── Matrix math ───────────────────────────────────────────────────────────────
const mmul = (A,B) => A.map(rA => B[0].map((_,j) => rA.reduce((s,v,k)=>s+v*B[k][j],0)));
const mT   = A => A[0].map((_,j) => A.map(r=>r[j]));
function minv(M) {
  const n=M.length, A=M.map((r,i)=>[...r,...Array.from({length:n},(_,j)=>i===j?1:0)]);
  for(let c=0;c<n;c++){
    let mx=c; for(let r=c+1;r<n;r++) if(Math.abs(A[r][c])>Math.abs(A[mx][c])) mx=r;
    [A[c],A[mx]]=[A[mx],A[c]];
    const p=A[c][c]||1e-14; for(let j=0;j<2*n;j++) A[c][j]/=p;
    for(let r=0;r<n;r++) if(r!==c){const f=A[r][c]; for(let j=0;j<2*n;j++) A[r][j]-=f*A[c][j];}
  }
  return A.map(r=>r.slice(n));
}
const vMean = a => a.reduce((s,v)=>s+v,0)/a.length;
const vStd  = (a,m) => { const mn=m??vMean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-mn)**2,0)/a.length)||1; };

// ── OLS + Analytical SHAP ─────────────────────────────────────────────────────
function computeModel(Xraw, y) {
  const n=Xraw.length, p=Xraw[0].length;
  const means = Xraw[0].map((_,j)=>vMean(Xraw.map(r=>r[j])));
  const stds  = means.map((m,j)=>vStd(Xraw.map(r=>r[j]),m));
  const Xsc   = Xraw.map(r=>r.map((v,j)=>(v-means[j])/stds[j]));
  const Xaug  = Xsc.map(r=>[1,...r]);
  const Xt=mT(Xaug); let XtX=mmul(Xt,Xaug);
  for(let i=1;i<=p;i++) XtX[i][i]+=0.001; // ridge
  const Xty=Xt.map(r=>r.reduce((s,v,i)=>s+v*y[i],0));
  const beta=mmul(minv(XtX),Xty.map(v=>[v])).map(r=>r[0]);
  const intercept=beta[0], coefs=beta.slice(1);
  const yPred=Xaug.map(r=>r.reduce((s,v,i)=>s+v*beta[i],0));
  const ym=vMean(y), ssR=y.reduce((s,v,i)=>s+(v-yPred[i])**2,0), ssT=y.reduce((s,v)=>s+(v-ym)**2,0);
  const r2=1-ssR/ssT, adjR2=1-(1-r2)*(n-1)/(n-p-1), rmse=Math.sqrt(ssR/n);
  const shapVals=Xsc.map(r=>r.map((v,j)=>v*coefs[j]));
  const meanAbs=coefs.map((_,j)=>vMean(shapVals.map(r=>Math.abs(r[j]))));
  const orderImp=[...meanAbs.map((v,i)=>({v,i}))].sort((a,b)=>b.v-a.v).map(d=>d.i);
  return { intercept,coefs,r2,adjR2,rmse,yPred,shapVals,meanAbs,orderImp,Xsc,Xraw,means,stds,n,p,ym };
}

// ── Color maps ─────────────────────────────────────────────────────────────────
function viridis(t) {
  const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  t=Math.max(0,Math.min(1,t)); const idx=t*(s.length-1),lo=Math.floor(idx),hi=Math.min(lo+1,s.length-1),f=idx-lo;
  return `rgb(${Math.round(s[lo][0]+f*(s[hi][0]-s[lo][0]))},${Math.round(s[lo][1]+f*(s[hi][1]-s[lo][1]))},${Math.round(s[lo][2]+f*(s[hi][2]-s[lo][2]))})`;
}
function rwb(t) {
  t=Math.max(0,Math.min(1,t));
  if(t<0.5){const f=t/0.5; return `rgb(${Math.round(31+f*224)},${Math.round(119+f*136)},${Math.round(180+f*75)})`;}
  else{const f=(t-0.5)/0.5; return `rgb(255,${Math.round(255-f*216)},${Math.round(255-f*215)})`;}
}
function normVal(v,mn,mx){return mx===mn?0.5:(v-mn)/(mx-mn);}

// ── SVG Download ───────────────────────────────────────────────────────────────
function dlSVG(el,name){
  if(!el) return;
  const svg=el.tagName==="svg"?el:el.querySelector("svg");
  if(!svg) return;
  const clone=svg.cloneNode(true);
  clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
  const bg=document.createElementNS("http://www.w3.org/2000/svg","rect");
  bg.setAttribute("width","100%"); bg.setAttribute("height","100%"); bg.setAttribute("fill","white");
  clone.insertBefore(bg,clone.firstChild);
  const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:"image/svg+xml"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name+".svg"; a.click();
}

// ── Upload Panel ───────────────────────────────────────────────────────────────
function UploadPanel({onData,xlsxReady}){
  const [drag,setDrag]=useState(false);
  const ref=useRef();
  const process=useCallback(async(file)=>{
    if(!file) return;
    const ext=file.name.split(".").pop().toLowerCase();
    let rows;
    if(ext==="csv"){
      const text=await file.text();
      const lines=text.trim().split(/\r?\n/);
      const headers=lines[0].split(",").map(h=>h.trim().replace(/^"|"$/g,""));
      rows=lines.slice(1).filter(l=>l.trim()).map(line=>{
        const vals=line.split(",").map(v=>v.trim().replace(/^"|"$/g,""));
        const o={}; headers.forEach((h,i)=>{ o[h]=isNaN(vals[i])||vals[i]===""?vals[i]:parseFloat(vals[i]); });
        return o;
      });
    } else if(["xlsx","xls"].includes(ext) && window.XLSX){
      const buf=await file.arrayBuffer();
      const wb=window.XLSX.read(buf,{type:"array"});
      rows=window.XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]],{defval:null});
    } else { alert("Please upload CSV or XLSX. Ensure file is valid."); return; }
    if(rows?.length>0) onData(rows,file.name);
  },[onData]);
  return(
    <div style={{border:`2px dashed ${drag?C.blue:C.border}`,borderRadius:10,padding:"28px 20px",textAlign:"center",background:drag?"#EAF4FB":C.paper,cursor:"pointer",transition:"all 0.2s"}}
      onClick={()=>ref.current.click()}
      onDragOver={e=>{e.preventDefault();setDrag(true);}}
      onDragLeave={()=>setDrag(false)}
      onDrop={e=>{e.preventDefault();setDrag(false);process(e.dataTransfer.files[0]);}}>
      <input ref={ref} type="file" accept=".csv,.xlsx,.xls" style={{display:"none"}} onChange={e=>process(e.target.files[0])}/>
      <div style={{fontSize:36,marginBottom:8}}>📂</div>
      <div style={{fontWeight:700,color:C.navy,fontSize:14,marginBottom:4}}>Drop file or click to upload</div>
      <div style={{fontSize:12,color:C.gray}}>CSV · XLSX · XLS</div>
    </div>
  );
}

// ── Figure title bar ───────────────────────────────────────────────────────────
function FigTitle({num,title,desc}){
  return(
    <div style={{marginBottom:12}}>
      <div style={{display:"flex",alignItems:"baseline",gap:8}}>
        <span style={{fontFamily:"Georgia,serif",fontWeight:700,color:C.navy,fontSize:13}}>Figure {num}.</span>
        <span style={{fontFamily:"Georgia,serif",fontWeight:600,color:C.ink,fontSize:13}}>{title}</span>
      </div>
      {desc&&<div style={{fontSize:11,color:C.gray,marginTop:3,fontStyle:"italic",fontFamily:"Georgia,serif"}}>{desc}</div>}
    </div>
  );
}

// ── Chart 1: Feature Importance ────────────────────────────────────────────────
function ImportanceBar({features,meanAbs,coefs,orderImp,target}){
  const ref=useRef();
  const sorted=[...orderImp].reverse();
  const W=560,bH=30,pL=178,pR=90,pT=28,pB=44;
  const H=sorted.length*bH+pT+pB;
  const maxV=Math.max(...meanAbs)*1.22;
  const sx=v=>pL+(v/maxV)*(W-pL-pR);
  const ticks=[0,0.25,0.5,0.75,1].map(t=>({t,x:sx(t*maxV),v:(t*maxV).toFixed(2)}));
  return(
    <div>
      <FigTitle num={1} title="SHAP Feature Importance" desc={`Mean absolute SHAP value for each feature predicting ${target}. Red = net positive effect on prediction; Blue = net negative.`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {sorted.map((fi,i)=>{
            const y0=pT+i*bH, fn=features[fi], v=meanAbs[fi], cl=coefs[fi]>=0?C.posShap:C.negShap, bw=sx(v)-pL;
            return(<g key={fi}>
              <text x={pL-8} y={y0+bH/2+4} textAnchor="end" fontSize={11} fill={C.navy} fontFamily="Georgia,serif">{fn.length>25?fn.slice(0,23)+"…":fn}</text>
              <rect x={pL} y={y0+5} width={Math.max(bw,2)} height={bH-10} fill={cl} opacity={0.84} rx={3}/>
              <text x={pL+bw+5} y={y0+bH/2+4} fontSize={10} fill={C.gray} fontFamily="monospace">{v.toFixed(3)}</text>
            </g>);
          })}
          <line x1={pL} y1={pT+sorted.length*bH} x2={W-pR} y2={pT+sorted.length*bH} stroke={C.border} strokeWidth={1}/>
          {ticks.map(({t,x,v})=><g key={t}>
            <line x1={x} y1={pT+sorted.length*bH} x2={x} y2={pT+sorted.length*bH+5} stroke={C.gray} strokeWidth={0.8}/>
            <text x={x} y={pT+sorted.length*bH+15} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v}</text>
          </g>)}
          <text x={pL+(W-pL-pR)/2} y={H-5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">Mean |SHAP value|</text>
          <rect x={W-pR+5} y={pT} width={11} height={11} fill={C.posShap} opacity={0.84} rx={1}/>
          <text x={W-pR+19} y={pT+9} fontSize={9} fill={C.gray}>Positive</text>
          <rect x={W-pR+5} y={pT+16} width={11} height={11} fill={C.negShap} opacity={0.84} rx={1}/>
          <text x={W-pR+19} y={pT+25} fontSize={9} fill={C.gray}>Negative</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig1_SHAP_Importance")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 2: Beeswarm ─────────────────────────────────────────────────────────
function BeeswarmPlot({features,shapVals,Xraw,orderImp,target}){
  const ref=useRef();
  const n=shapVals.length, p=features.length;
  const W=580, pH=32, pL=175, pR=50, pT=28, pB=44;
  const H=p*pH+pT+pB;
  const allSV=shapVals.flatMap(r=>r); const mnSV=Math.min(...allSV), mxSV=Math.max(...allSV);
  const svRange=Math.max(Math.abs(mnSV),Math.abs(mxSV));
  const innerW=W-pL-pR;
  const sx=v=>pL+innerW/2+v/svRange*(innerW/2)*0.92;
  const rng=(seed)=>{ // seeded random
    let s=seed; return ()=>{ s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; };
  };
  const cbW=14, cbH=Math.min(p*pH-8, 80);
  const cbX=W-pR-cbW-4, cbY=pT+(p*pH-cbH)/2;
  return(
    <div>
      <FigTitle num={2} title="SHAP Beeswarm Summary" desc={`Each dot is one observation. Horizontal position = SHAP value (impact on ${target}). Colour = feature value magnitude (low → blue → yellow → high).`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {/* Zero line */}
          <line x1={pL+innerW/2} y1={pT} x2={pL+innerW/2} y2={pT+p*pH} stroke={C.border} strokeWidth={1} strokeDasharray="3,3"/>
          {orderImp.map((fi,row)=>{
            const y0=pT+row*pH+pH/2;
            const sv=shapVals.map(r=>r[fi]);
            const fv=Xraw.map(r=>r[fi]);
            const fvMn=Math.min(...fv), fvMx=Math.max(...fv);
            const rand=rng(fi*137);
            const sorted=[...sv.map((v,i)=>({v,fv:fv[i],i}))].sort((a,b)=>a.v-b.v);
            return(<g key={fi}>
              <text x={pL-8} y={y0+4} textAnchor="end" fontSize={11} fill={C.navy} fontFamily="Georgia,serif">
                {features[fi].length>24?features[fi].slice(0,22)+"…":features[fi]}
              </text>
              {sorted.map(({v,fv:fval},si)=>{
                const cx=sx(v), jit=(rand()-0.5)*pH*0.7;
                const col=viridis(normVal(fval,fvMn,fvMx));
                return <circle key={si} cx={cx} cy={y0+jit} r={3.5} fill={col} opacity={0.85} stroke="white" strokeWidth={0.5}/>;
              })}
            </g>);
          })}
          {/* X axis */}
          <line x1={pL} y1={pT+p*pH} x2={W-pR-cbW-12} y2={pT+p*pH} stroke={C.border} strokeWidth={1}/>
          {[-1,-0.5,0,0.5,1].map(t=>{
            const x=sx(t*svRange), v=(t*svRange).toFixed(2);
            return(<g key={t}>
              <line x1={x} y1={pT+p*pH} x2={x} y2={pT+p*pH+5} stroke={C.gray} strokeWidth={0.8}/>
              <text x={x} y={pT+p*pH+15} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v}</text>
            </g>);
          })}
          <text x={pL+innerW/2-20} y={H-5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">SHAP value (impact on model output)</text>
          {/* Colorbar */}
          {Array.from({length:cbH},(_,k)=>{
            const t=1-k/cbH; return <rect key={k} x={cbX} y={cbY+k} width={cbW} height={1.5} fill={viridis(t)}/>;
          })}
          <rect x={cbX} y={cbY} width={cbW} height={cbH} fill="none" stroke={C.border} strokeWidth={0.5}/>
          <text x={cbX+cbW+3} y={cbY+6} fontSize={8} fill={C.gray}>High</text>
          <text x={cbX+cbW+3} y={cbY+cbH+4} fontSize={8} fill={C.gray}>Low</text>
          <text x={cbX+cbW/2} y={cbY-4} textAnchor="middle" fontSize={8} fill={C.gray}>Feature</text>
          <text x={cbX+cbW/2} y={cbY-14} textAnchor="middle" fontSize={8} fill={C.gray}>value</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig2_SHAP_Beeswarm")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 3: Dependence Plots ─────────────────────────────────────────────────
function DependencePlots({features,shapVals,Xraw,orderImp,target}){
  const ref=useRef();
  const top4=orderImp.slice(0,4);
  const cols=2, rows=2;
  const cW=260, cH=200, pad={t:30,r:15,b:44,l:52}, gap=20;
  const W=cols*cW+gap, H=rows*cH+gap;
  const allColors=Xraw.map(r=>r[orderImp[0]]);
  const acMin=Math.min(...allColors), acMax=Math.max(...allColors);
  return(
    <div>
      <FigTitle num={3} title="SHAP Dependence Plots — Top 4 Features" desc={`Feature value (x) vs. SHAP contribution (y) for the four most influential features. Colour indicates feature value magnitude.`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {top4.map((fi,idx)=>{
            const col=idx%2, row=Math.floor(idx/2);
            const ox=col*(cW+gap), oy=row*(cH+gap);
            const iW=cW-pad.l-pad.r, iH=cH-pad.t-pad.b;
            const fvArr=Xraw.map(r=>r[fi]);
            const svArr=shapVals.map(r=>r[fi]);
            const fvMin=Math.min(...fvArr), fvMax=Math.max(...fvArr);
            const svMin=Math.min(...svArr), svMax=Math.max(...svArr);
            const svPad=(svMax-svMin)*0.12||0.5;
            const sx2=v=>ox+pad.l+(v-fvMin)/(fvMax-fvMin||1)*iW;
            const sy2=v=>oy+pad.t+iH-(v-(svMin-svPad))/((svMax+svPad)-(svMin-svPad))*iH;
            // trend line
            const n=fvArr.length;
            const xm=vMean(fvArr), ym2=vMean(svArr);
            const slope=fvArr.reduce((s,v,i)=>s+(v-xm)*(svArr[i]-ym2),0)/(fvArr.reduce((s,v)=>s+(v-xm)**2,0)||1);
            const intcpt=ym2-slope*xm;
            const x1l=fvMin, x2l=fvMax, y1l=slope*x1l+intcpt, y2l=slope*x2l+intcpt;
            return(<g key={fi}>
              {/* Background */}
              <rect x={ox+pad.l} y={oy+pad.t} width={iW} height={iH} fill={C.bg} rx={2}/>
              {/* Zero line */}
              {svMin<0&&svMax>0&&<line x1={ox+pad.l} y1={sy2(0)} x2={ox+pad.l+iW} y2={sy2(0)} stroke={C.border} strokeWidth={0.8} strokeDasharray="3,2"/>}
              {/* Trend line */}
              <line x1={sx2(x1l)} y1={sy2(y1l)} x2={sx2(x2l)} y2={sy2(y2l)} stroke={C.ink} strokeWidth={1.3} strokeDasharray="4,3" opacity={0.6}/>
              {/* Points */}
              {fvArr.map((fv,i)=><circle key={i} cx={sx2(fv)} cy={sy2(svArr[i])} r={3.5} fill={viridis(normVal(fv,fvMin,fvMax))} opacity={0.8} stroke="white" strokeWidth={0.4}/>)}
              {/* Axes */}
              <line x1={ox+pad.l} y1={oy+pad.t+iH} x2={ox+pad.l+iW} y2={oy+pad.t+iH} stroke={C.border} strokeWidth={1}/>
              <line x1={ox+pad.l} y1={oy+pad.t} x2={ox+pad.l} y2={oy+pad.t+iH} stroke={C.border} strokeWidth={1}/>
              {/* Labels */}
              <text x={ox+pad.l+iW/2} y={oy+cH-4} textAnchor="middle" fontSize={10} fill={C.gray} fontFamily="Georgia,serif">{features[fi].length>20?features[fi].slice(0,18)+"…":features[fi]}</text>
              <text x={ox+8} y={oy+pad.t+iH/2} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="Georgia,serif" transform={`rotate(-90,${ox+8},${oy+pad.t+iH/2})`}>SHAP</text>
              <text x={ox+pad.l+iW/2} y={oy+pad.t-8} textAnchor="middle" fontSize={11} fill={C.ink} fontWeight="bold" fontFamily="Georgia,serif">{features[fi].length>20?features[fi].slice(0,18)+"…":features[fi]}</text>
              {/* Y tick marks */}
              {[svMin,ym2,svMax].map((v,k)=><g key={k}>
                <line x1={ox+pad.l-3} y1={sy2(v)} x2={ox+pad.l} y2={sy2(v)} stroke={C.gray} strokeWidth={0.7}/>
                <text x={ox+pad.l-5} y={sy2(v)+3} textAnchor="end" fontSize={8} fill={C.gray} fontFamily="monospace">{v.toFixed(1)}</text>
              </g>)}
            </g>);
          })}
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig3_SHAP_Dependence")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 4: Waterfall ────────────────────────────────────────────────────────
function WaterfallPlot({features,shapVals,Xraw,y,yPred,orderImp,intercept,target}){
  const ref=useRef();
  const n=shapVals.length, p=features.length;
  // Pick low / mid / high prediction observations
  const sorted=[...yPred.map((v,i)=>({v,i}))].sort((a,b)=>a.v-b.v);
  const picks=[sorted[0].i, sorted[Math.floor(n/2)].i, sorted[n-1].i];
  const labels=["Minimum prediction","Median prediction","Maximum prediction"];
  const panels=3, pW=220, pH=280, gap=20, padL=100, padR=50, padT=44, padB=36;
  const W=panels*(pW+gap)-gap, H=pH;
  const iW=pW-padL-padR;
  return(
    <div>
      <FigTitle num={4} title="SHAP Waterfall Plots" desc={`Decomposition of individual predictions into feature contributions. Each bar shows how a feature pushes the prediction above (red) or below (blue) the baseline E[ŷ] = ${intercept.toFixed(2)}.`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {picks.map((obsIdx,panelIdx)=>{
            const ox=panelIdx*(pW+gap);
            const sv=shapVals[obsIdx];
            // Sort by |SHAP| ascending (smallest at top, biggest at bottom)
            const ord=[...orderImp].reverse(); // shows least important at top, most at bottom
            const iH=pH-padT-padB;
            const bH=Math.floor(iH/p)-2;
            const allVals=[...sv.map((v,j)=>intercept+sv.slice(0,j+1).reduce((s,_,k)=>s+sv[orderImp[k]],0))];
            let cumulative=intercept;
            const allX=[intercept,...ord.map(fi=>{ cumulative+=sv[fi]; return cumulative; })];
            const allV=[...ord.map(fi=>sv[fi])];
            const xMin=Math.min(intercept,...allX)*0.97, xMax=Math.max(intercept,...allX)*1.03;
            const sx2=v=>ox+padL+(v-xMin)/(xMax-xMin)*iW;
            const predV=yPred[obsIdx], actV=y[obsIdx];
            let run=intercept;
            return(<g key={obsIdx}>
              {/* Title */}
              <text x={ox+pW/2} y={20} textAnchor="middle" fontSize={11} fontWeight="bold" fill={C.ink} fontFamily="Georgia,serif">{labels[panelIdx]}</text>
              <text x={ox+pW/2} y={33} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">ŷ={predV.toFixed(2)} | y={actV.toFixed(2)}</text>
              {ord.map((fi,i)=>{
                const v=sv[fi], barStart=run, y0=padT+i*(iH/p);
                const color=v>=0?C.posShap:C.negShap;
                const bw=(v/(xMax-xMin))*iW, bx=sx2(Math.min(barStart,barStart+v));
                run+=v;
                const fn=features[fi]; const fnS=fn.length>14?fn.slice(0,12)+"…":fn;
                return(<g key={fi}>
                  <text x={ox+padL-6} y={y0+iH/p/2+4} textAnchor="end" fontSize={9} fill={C.navy} fontFamily="Georgia,serif">{fnS}</text>
                  <rect x={bx} y={y0+2} width={Math.max(Math.abs(bw),1.5)} height={iH/p-4} fill={color} opacity={0.82} rx={2}/>
                  <text x={v>=0?ox+padL+((run-xMin)/(xMax-xMin))*iW+3:ox+padL+((run-xMin)/(xMax-xMin))*iW-3}
                    y={y0+iH/p/2+3.5} fontSize={8} fill={C.gray} fontFamily="monospace"
                    textAnchor={v>=0?"start":"end"}>{v>0?"+":""}{v.toFixed(2)}</text>
                </g>);
              })}
              {/* Baseline and prediction lines */}
              <line x1={sx2(intercept)} y1={padT} x2={sx2(intercept)} y2={padT+iH} stroke={C.gray} strokeWidth={0.8} strokeDasharray="2,2"/>
              <line x1={sx2(predV)} y1={padT} x2={sx2(predV)} y2={padT+iH} stroke={C.ink} strokeWidth={1.2} strokeDasharray="4,2"/>
              {/* X axis */}
              <line x1={ox+padL} y1={padT+iH} x2={ox+padL+iW} y2={padT+iH} stroke={C.border} strokeWidth={1}/>
              {[xMin,(xMin+xMax)/2,xMax].map((v,k)=><g key={k}>
                <line x1={sx2(v)} y1={padT+iH} x2={sx2(v)} y2={padT+iH+4} stroke={C.gray} strokeWidth={0.7}/>
                <text x={sx2(v)} y={padT+iH+13} textAnchor="middle" fontSize={8} fill={C.gray} fontFamily="monospace">{v.toFixed(1)}</text>
              </g>)}
              <text x={ox+padL+iW/2} y={pH-4} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="Georgia,serif">{target}</text>
            </g>);
          })}
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig4_SHAP_Waterfall")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 5: Heatmap ──────────────────────────────────────────────────────────
function HeatmapPlot({features,shapVals,yPred,orderImp,target}){
  const ref=useRef();
  const n=shapVals.length, p=features.length;
  const sortByPred=[...yPred.map((v,i)=>({v,i}))].sort((a,b)=>a.v-b.v).map(d=>d.i);
  const W=Math.min(700,Math.max(480,n*11+200)), cellW=Math.max(4,(W-200)/n);
  const cellH=20, pL=175, pT=50, pB=52, pR=60;
  const H=p*cellH+pT+pB;
  const allSV=shapVals.flatMap(r=>r);
  const svAbs=Math.max(Math.abs(Math.min(...allSV)),Math.abs(Math.max(...allSV)));
  const cbH=Math.min(p*cellH*0.8,100), cbW=12, cbX=W-pR+8, cbY=pT+(p*cellH-cbH)/2;
  return(
    <div>
      <FigTitle num={5} title="SHAP Value Heatmap" desc={`Observations (x-axis, sorted by predicted ${target}) × features (y-axis, sorted by importance). Colour encodes SHAP value: red = positive contribution, blue = negative.`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {/* Prediction bar at top */}
          {sortByPred.map((obsI,si)=>{
            const pv=yPred[obsI], pvMin=Math.min(...yPred), pvMax=Math.max(...yPred);
            return <rect key={si} x={pL+si*cellW} y={pT-20} width={cellW+0.5} height={10}
              fill={viridis(normVal(pv,pvMin,pvMax))} opacity={0.9}/>;
          })}
          <text x={pL-6} y={pT-13} textAnchor="end" fontSize={9} fill={C.gray} fontFamily="Georgia,serif">ŷ</text>
          {/* Cells */}
          {orderImp.map((fi,row)=>{
            const y0=pT+row*cellH;
            return(<g key={fi}>
              <text x={pL-6} y={y0+cellH/2+4} textAnchor="end" fontSize={10} fill={C.navy} fontFamily="Georgia,serif">
                {features[fi].length>24?features[fi].slice(0,22)+"…":features[fi]}
              </text>
              {sortByPred.map((obsI,col)=>{
                const sv=shapVals[obsI][fi];
                const t=0.5+sv/svAbs/2;
                return <rect key={col} x={pL+col*cellW} y={y0+1} width={cellW+0.3} height={cellH-2} fill={rwb(t)}/>;
              })}
            </g>);
          })}
          {/* Border */}
          <rect x={pL} y={pT} width={n*cellW} height={p*cellH} fill="none" stroke={C.border} strokeWidth={0.8}/>
          {/* X axis label */}
          <text x={pL+n*cellW/2} y={H-5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">
            Observations (sorted by predicted {target})
          </text>
          {/* Colorbar */}
          {Array.from({length:Math.ceil(cbH)},(_,k)=>{
            const t=1-k/cbH; return <rect key={k} x={cbX} y={cbY+k} width={cbW} height={1.5} fill={rwb(t)}/>;
          })}
          <rect x={cbX} y={cbY} width={cbW} height={cbH} fill="none" stroke={C.border} strokeWidth={0.5}/>
          <text x={cbX+cbW+3} y={cbY+6} fontSize={8} fill={C.posShap} fontWeight="bold">+</text>
          <text x={cbX+cbW+3} y={cbY+cbH+4} fontSize={8} fill={C.negShap} fontWeight="bold">−</text>
          <text x={cbX+cbW/2} y={cbY-4} textAnchor="middle" fontSize={8} fill={C.gray}>SHAP</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig5_SHAP_Heatmap")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 6: Force Stacked ────────────────────────────────────────────────────
function ForceStacked({features,shapVals,yPred,orderImp,intercept,target}){
  const ref=useRef();
  const n=shapVals.length, p=features.length;
  const sortI=[...yPred.map((v,i)=>({v,i}))].sort((a,b)=>a.v-b.v).map(d=>d.i);
  const W=Math.min(720,Math.max(520,n*9+160)), barW=Math.max(3,(W-180)/n);
  const pL=65, pR=20, pT=30, pB=44, iH=180;
  const H=iH+pT+pB;
  const yMin=Math.min(...yPred)*0.96, yMax=Math.max(...yPred)*1.04;
  const sy=v=>pT+iH-(v-yMin)/(yMax-yMin)*iH;
  const yticks=[yMin,(yMin+yMax)/2,yMax];
  return(
    <div>
      <FigTitle num={6} title="SHAP Force Plot — All Observations" desc={`Stacked SHAP contributions for each observation (sorted by predicted value). Each colour represents a feature's additive contribution to the final prediction.`}/>
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",maxWidth:W,display:"block"}} xmlns="http://www.w3.org/2000/svg">
          {sortI.map((obsI,si)=>{
            const x=pL+si*barW;
            // Separate pos and neg SHAP
            let posTop=intercept, negBot=intercept;
            return(<g key={obsI}>
              {orderImp.map((fi,k)=>{
                const sv=shapVals[obsI][fi];
                if(sv>=0){
                  const y1=sy(posTop+sv), y2=sy(posTop);
                  posTop+=sv;
                  return <rect key={fi} x={x} y={y1} width={barW} height={Math.max(y2-y1,0.5)} fill={TAB_COLS[k%TAB_COLS.length]} opacity={0.85}/>;
                } else {
                  const y1=sy(negBot), y2=sy(negBot+sv);
                  negBot+=sv;
                  return <rect key={fi} x={x} y={y1} width={barW} height={Math.max(y2-y1,0.5)} fill={TAB_COLS[k%TAB_COLS.length]} opacity={0.85}/>;
                }
              })}
            </g>);
          })}
          {/* Baseline */}
          <line x1={pL} y1={sy(intercept)} x2={pL+n*barW} y2={sy(intercept)} stroke={C.ink} strokeWidth={1.2} strokeDasharray="4,3" opacity={0.7}/>
          <text x={pL-4} y={sy(intercept)+3} textAnchor="end" fontSize={9} fill={C.ink} fontFamily="monospace">E[ŷ]={intercept.toFixed(1)}</text>
          {/* Y axis */}
          <line x1={pL} y1={pT} x2={pL} y2={pT+iH} stroke={C.border} strokeWidth={1}/>
          {yticks.map((v,k)=><g key={k}>
            <line x1={pL-3} y1={sy(v)} x2={pL} y2={sy(v)} stroke={C.gray} strokeWidth={0.7}/>
            <text x={pL-5} y={sy(v)+3} textAnchor="end" fontSize={9} fill={C.gray} fontFamily="monospace">{v.toFixed(1)}</text>
          </g>)}
          {/* X axis */}
          <line x1={pL} y1={pT+iH} x2={pL+n*barW} y2={pT+iH} stroke={C.border} strokeWidth={1}/>
          <text x={pL+n*barW/2} y={H-5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">Observations (sorted by ŷ)</text>
          <text x={14} y={pT+iH/2} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif" transform={`rotate(-90,14,${pT+iH/2})`}>{target}</text>
          {/* Legend (top-5 features) */}
          {orderImp.slice(0,5).map((fi,k)=><g key={fi}>
            <rect x={pL+n*barW+5} y={pT+k*16} width={10} height={10} fill={TAB_COLS[k%TAB_COLS.length]} opacity={0.85} rx={1}/>
            <text x={pL+n*barW+18} y={pT+k*16+9} fontSize={9} fill={C.gray}>{features[fi].length>15?features[fi].slice(0,13)+"…":features[fi]}</text>
          </g>)}
        </svg>
      </div>
      <button style={dlBtn} onClick={()=>dlSVG(ref.current,"Fig6_SHAP_Force")}>↓ Download SVG</button>
    </div>
  );
}

// ── Chart 7: Model Summary ────────────────────────────────────────────────────
function ModelSummary({features,coefs,intercept,r2,adjR2,rmse,meanAbs,orderImp,n,p,target}){
  return(
    <div>
      <FigTitle num={7} title="Model & SHAP Summary Statistics" desc={`OLS regression summary for predicting ${target}. SHAP values are analytical (exact) for linear models.`}/>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
        {[["R²",r2.toFixed(4)],["Adj. R²",adjR2.toFixed(4)],["RMSE",rmse.toFixed(4)],["n",n],["Features",p],["Baseline E[ŷ]",intercept.toFixed(4)]]
          .map(([k,v])=>(
            <div key={k} style={{background:C.bg,borderRadius:6,padding:"10px 14px",border:`1px solid ${C.border}`}}>
              <div style={{fontSize:11,color:C.gray,fontFamily:"Georgia,serif",marginBottom:2}}>{k}</div>
              <div style={{fontSize:18,fontWeight:700,color:C.navy,fontFamily:"monospace"}}>{v}</div>
            </div>
          ))}
      </div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:12,fontFamily:"Georgia,serif"}}>
          <thead>
            <tr style={{background:C.navy,color:"white"}}>
              {["Rank","Feature","Coefficient","Mean |SHAP|","Direction"].map(h=>(
                <th key={h} style={{padding:"7px 10px",textAlign:"left",fontWeight:600,fontSize:11}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {orderImp.map((fi,k)=>(
              <tr key={fi} style={{background:k%2===0?C.paper:C.bg}}>
                <td style={{padding:"6px 10px",color:C.gray}}>{k+1}</td>
                <td style={{padding:"6px 10px",fontWeight:600,color:C.navy}}>{features[fi]}</td>
                <td style={{padding:"6px 10px",fontFamily:"monospace",color:coefs[fi]>=0?C.posShap:C.negShap}}>{coefs[fi].toFixed(4)}</td>
                <td style={{padding:"6px 10px",fontFamily:"monospace"}}>{meanAbs[fi].toFixed(4)}</td>
                <td style={{padding:"6px 10px"}}>
                  <span style={{background:coefs[fi]>=0?"#FDECEA":"#EBF5FB",color:coefs[fi]>=0?C.posShap:C.negShap,padding:"2px 8px",borderRadius:10,fontSize:10,fontWeight:600}}>
                    {coefs[fi]>=0?"▲ Positive":"▼ Negative"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App(){
  const xlsxReady=useXLSX();
  const [rawData,setRawData]=useState(null);
  const [fileName,setFileName]=useState("");
  const [columns,setColumns]=useState([]);
  const [targetCol,setTargetCol]=useState("");
  const [model,setModel]=useState(null);
  const [activeTab,setActiveTab]=useState("importance");
  const [loading,setLoading]=useState(false);

  const handleData=useCallback((rows,fname)=>{
    setRawData(rows); setFileName(fname);
    const cols=Object.keys(rows[0]).filter(k=>typeof rows[0][k]==="number"&&rows.every(r=>typeof r[k]==="number"));
    setColumns(cols);
    setTargetCol(cols[cols.length-1]||"");
    setModel(null);
  },[]);

  const run=useCallback(()=>{
    if(!rawData||!targetCol) return;
    setLoading(true);
    setTimeout(()=>{
      try{
        const featureNames=columns.filter(c=>c!==targetCol);
        const Xraw=rawData.map(r=>featureNames.map(f=>r[f]));
        const y=rawData.map(r=>r[targetCol]);
        const m=computeModel(Xraw,y);
        setModel({...m,featureNames,target:targetCol,y,rawData});
      }catch(e){alert("Model error: "+e.message);}
      setLoading(false);
    },50);
  },[rawData,targetCol,columns]);

  const tabs=[
    {id:"importance",label:"① Importance"},
    {id:"beeswarm",label:"② Beeswarm"},
    {id:"dependence",label:"③ Dependence"},
    {id:"waterfall",label:"④ Waterfall"},
    {id:"heatmap",label:"⑤ Heatmap"},
    {id:"force",label:"⑥ Force Plot"},
    {id:"summary",label:"⑦ Summary"},
  ];

  return(
    <div style={{minHeight:"100vh",background:C.bg,fontFamily:"Georgia,serif"}}>
      {/* Header */}
      <div style={{background:C.navy,padding:"14px 32px",display:"flex",alignItems:"center",justifyContent:"space-between",boxShadow:"0 2px 8px rgba(0,0,0,0.15)"}}>
        <div>
          <div style={{color:"white",fontSize:18,fontWeight:700,letterSpacing:0.3}}>SHAP Analysis Studio</div>
          <div style={{color:"#A5C8E1",fontSize:11,marginTop:1}}>Publication-quality explainability plots</div>
        </div>
        {model&&<div style={{color:"#A5C8E1",fontSize:12,fontFamily:"monospace"}}>
          {fileName} · n={model.n} · p={model.p} · R²={model.r2.toFixed(3)}
        </div>}
      </div>

      <div style={{maxWidth:1100,margin:"0 auto",padding:"24px 20px",display:"grid",gridTemplateColumns:model?"260px 1fr":"1fr",gap:24,alignItems:"start"}}>
        {/* Sidebar */}
        <div>
          <div style={cardStyle}>
            <div style={{fontWeight:700,color:C.navy,marginBottom:12,fontSize:13}}>1 · Upload Data</div>
            <UploadPanel onData={handleData} xlsxReady={xlsxReady}/>
            {fileName&&<div style={{fontSize:11,color:C.gray,marginTop:8,fontFamily:"monospace",wordBreak:"break-all"}}>📄 {fileName}</div>}
          </div>

          {columns.length>0&&(
            <div style={cardStyle}>
              <div style={{fontWeight:700,color:C.navy,marginBottom:10,fontSize:13}}>2 · Select Target Variable</div>
              <select value={targetCol} onChange={e=>setTargetCol(e.target.value)}
                style={{width:"100%",padding:"7px 10px",borderRadius:5,border:`1px solid ${C.border}`,fontSize:12,background:C.paper,color:C.navy,fontFamily:"Georgia,serif"}}>
                {columns.map(c=><option key={c} value={c}>{c}</option>)}
              </select>
              <div style={{fontSize:11,color:C.gray,marginTop:6}}>All other numeric columns will be used as features.</div>
              <button onClick={run} disabled={loading||!targetCol}
                style={{marginTop:12,width:"100%",padding:"9px 0",background:loading?C.gray:C.blue,color:"white",border:"none",borderRadius:6,fontSize:13,fontWeight:700,cursor:loading?"not-allowed":"pointer",letterSpacing:0.3,fontFamily:"Georgia,serif",transition:"background 0.2s"}}>
                {loading?"Computing SHAP…":"Run SHAP Analysis"}
              </button>
            </div>
          )}

          {model&&(
            <div style={cardStyle}>
              <div style={{fontWeight:700,color:C.navy,marginBottom:8,fontSize:13}}>3 · Model Fit</div>
              {[["R²",model.r2.toFixed(4)],["Adj. R²",model.adjR2.toFixed(4)],["RMSE",model.rmse.toFixed(4)]].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:`1px solid ${C.light}`,fontSize:12}}>
                  <span style={{color:C.gray}}>{k}</span>
                  <span style={{fontFamily:"monospace",fontWeight:700,color:C.navy}}>{v}</span>
                </div>
              ))}
              <div style={{marginTop:8,fontSize:11,color:C.gray}}>OLS with ridge regularisation (λ=0.001). Analytical SHAP exact for linear models.</div>
            </div>
          )}
        </div>

        {/* Main Content */}
        {model?(
          <div>
            {/* Tab bar */}
            <div style={{display:"flex",gap:4,marginBottom:16,flexWrap:"wrap"}}>
              {tabs.map(tab=>(
                <button key={tab.id} onClick={()=>setActiveTab(tab.id)}
                  style={{padding:"7px 14px",fontSize:12,border:"none",borderRadius:20,cursor:"pointer",fontFamily:"Georgia,serif",fontWeight:activeTab===tab.id?700:400,
                    background:activeTab===tab.id?C.navy:C.paper,color:activeTab===tab.id?"white":C.gray,
                    boxShadow:activeTab===tab.id?"0 2px 6px rgba(0,0,0,0.15)":"0 1px 3px rgba(0,0,0,0.06)",transition:"all 0.15s"}}>
                  {tab.label}
                </button>
              ))}
            </div>

            <div style={cardStyle}>
              {activeTab==="importance"&&<ImportanceBar features={model.featureNames} meanAbs={model.meanAbs} coefs={model.coefs} orderImp={model.orderImp} target={model.target}/>}
              {activeTab==="beeswarm"&&<BeeswarmPlot features={model.featureNames} shapVals={model.shapVals} Xraw={model.Xraw} orderImp={model.orderImp} target={model.target}/>}
              {activeTab==="dependence"&&<DependencePlots features={model.featureNames} shapVals={model.shapVals} Xraw={model.Xraw} orderImp={model.orderImp} target={model.target}/>}
              {activeTab==="waterfall"&&<WaterfallPlot features={model.featureNames} shapVals={model.shapVals} Xraw={model.Xraw} y={model.y} yPred={model.yPred} orderImp={model.orderImp} intercept={model.intercept} target={model.target}/>}
              {activeTab==="heatmap"&&<HeatmapPlot features={model.featureNames} shapVals={model.shapVals} yPred={model.yPred} orderImp={model.orderImp} target={model.target}/>}
              {activeTab==="force"&&<ForceStacked features={model.featureNames} shapVals={model.shapVals} yPred={model.yPred} orderImp={model.orderImp} intercept={model.intercept} target={model.target}/>}
              {activeTab==="summary"&&<ModelSummary features={model.featureNames} coefs={model.coefs} intercept={model.intercept} r2={model.r2} adjR2={model.adjR2} rmse={model.rmse} meanAbs={model.meanAbs} orderImp={model.orderImp} n={model.n} p={model.p} target={model.target}/>}
            </div>

            <div style={{fontSize:11,color:C.gray,marginTop:8,fontStyle:"italic",textAlign:"right"}}>
              All plots export as publication-quality SVG vector graphics
            </div>
          </div>
        ):(
          <div style={{...cardStyle,display:"flex",alignItems:"center",justifyContent:"center",minHeight:320,flexDirection:"column",gap:12}}>
            <div style={{fontSize:48}}>📈</div>
            <div style={{fontWeight:700,color:C.navy,fontSize:16}}>Upload a dataset to begin</div>
            <div style={{color:C.gray,fontSize:13,maxWidth:360,textAlign:"center"}}>
              Upload any CSV or Excel file with numeric columns. Select a target variable, then run the SHAP analysis to generate publication-ready figures.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
