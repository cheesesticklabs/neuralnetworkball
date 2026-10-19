(function(root){
'use strict';
function random(seed){let v=seed>>>0;return()=>{v=(1664525*v+1013904223)>>>0;return v/4294967296}}
function label(x,y,rule='different'){const a=x>=.5,b=y>=.5;return rule==='different'?Number(a!==b):Number(a||b)}
const training=[[0,0],[0,1],[1,0],[1,1]];
const tests=[[.12,.08],[.85,.14],[.13,.88],[.91,.84],[.22,.17],[.78,.24],[.18,.76],[.82,.91]];
function make(size=4,seed=17){const r=random(seed),scale=1.5;return{size,epoch:0,w:Array.from({length:size},()=>[(r()-.5)*scale,(r()-.5)*scale]),b:Array.from({length:size},()=>(r()-.5)*scale),v:Array.from({length:size},()=>(r()-.5)*scale),c:(r()-.5)*scale}}
function forward(n,x,y){const h=n.w.map((w,j)=>Math.tanh(w[0]*x+w[1]*y+n.b[j]));const z=h.reduce((s,a,j)=>s+a*n.v[j],n.c);const p=1/(1+Math.exp(-Math.max(-40,Math.min(40,z))));return{h,z,p,prediction:Number(p>=.5)}}
function train(n,rule='different',epochs=100,rate=.15){for(let e=0;e<epochs;e++){for(let i=0;i<4;i++){const k=(i+n.epoch)%4,[x,y]=training[k],t=label(x,y,rule),f=forward(n,x,y),d=f.p-t;const dv=n.v.slice();for(let j=0;j<n.size;j++){const dh=d*dv[j]*(1-f.h[j]*f.h[j]);n.v[j]-=rate*d*f.h[j];n.w[j][0]-=rate*dh*x;n.w[j][1]-=rate*dh*y;n.b[j]-=rate*dh}n.c-=rate*d}n.epoch++}return n}
function metrics(n,points=training,rule='different'){let correct=0,loss=0;points.forEach(([x,y])=>{const f=forward(n,x,y),t=label(x,y,rule);correct+=Number(f.prediction===t);const p=Math.max(1e-10,Math.min(1-1e-10,f.p));loss-=t*Math.log(p)+(1-t)*Math.log(1-p)});return{correct,total:points.length,loss:loss/points.length}}
const threeTrain=[[0,0,0],[0,0,1],[0,1,0],[1,0,1],[1,1,0],[1,1,1]];
const threeTest=[[0,1,1],[1,0,0]];
const threeLabel=a=>Number(!a.every(v=>v===a[0]));
function makeThree(){const n=make(4,17),r=random(17);n.w=Array.from({length:4},()=>Array.from({length:3},()=>(r()-.5)*1.5));return n}
function forwardThree(n,a){const h=n.w.map((w,j)=>Math.tanh(w.reduce((v,q,i)=>v+q*a[i],n.b[j])));const z=h.reduce((v,q,j)=>v+q*n.v[j],n.c);const p=1/(1+Math.exp(-Math.max(-40,Math.min(40,z))));return{h,z,p,prediction:Number(p>=.5)}}
function trainThree(n,epochs=100){for(let e=0;e<epochs;e++){for(let i=0;i<threeTrain.length;i++){const a=threeTrain[(i+n.epoch)%threeTrain.length],t=threeLabel(a),f=forwardThree(n,a),d=f.p-t,old=n.v.slice();for(let j=0;j<n.size;j++){const dh=d*old[j]*(1-f.h[j]*f.h[j]);n.v[j]-=.15*d*f.h[j];for(let k=0;k<3;k++)n.w[j][k]-=.15*dh*a[k];n.b[j]-=.15*dh}n.c-=.15*d}n.epoch++}return n}
function scoreThree(n,data){return{correct:data.filter(a=>forwardThree(n,a).prediction===threeLabel(a)).length,total:data.length}}

const fourAll=Array.from({length:16},(_,i)=>[3,2,1,0].map(k=>(i>>k)&1));
const heldFour=new Set([3,5,10,12]);
const fourTrain=fourAll.filter((_,i)=>!heldFour.has(i));
const fourTest=fourAll.filter((_,i)=>heldFour.has(i));
function makeFour(){const n=make(4,17),r=random(17);n.w=Array.from({length:4},()=>Array.from({length:4},()=>(r()-.5)*1.5));return n}
function trainFour(n,epochs=100){for(let e=0;e<epochs;e++){for(let i=0;i<fourTrain.length;i++){const a=fourTrain[(i+n.epoch)%fourTrain.length],t=threeLabel(a),f=forwardThree(n,a),d=f.p-t,old=n.v.slice();for(let j=0;j<n.size;j++){const dh=d*old[j]*(1-f.h[j]*f.h[j]);n.v[j]-=.15*d*f.h[j];for(let k=0;k<4;k++)n.w[j][k]-=.15*dh*a[k];n.b[j]-=.15*dh}n.c-=.15*d}n.epoch++}return n}

root.Neural={random,label,training,tests,make,forward,train,metrics,threeTrain,threeTest,threeLabel,makeThree,forwardThree,trainThree,scoreThree,fourTrain,fourTest,fourLabel:threeLabel,makeFour,forwardFour:forwardThree,trainFour,scoreFour:scoreThree};
})(typeof module!=='undefined'?module.exports:window);
