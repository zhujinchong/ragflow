(self.webpackChunk=self.webpackChunk||[]).push([[4144],{63606:function(t,n,e){"use strict";e.d(n,{Z:function(){return v}});var o=e(87462),l=e(62435),f={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M912 190h-69.9c-9.8 0-19.1 4.5-25.1 12.2L404.7 724.5 207 474a32 32 0 00-25.1-12.2H112c-6.7 0-10.4 7.7-6.3 12.9l273.9 347c12.8 16.2 37.4 16.2 50.3 0l488.4-618.9c4.1-5.1.4-12.8-6.3-12.8z"}}]},name:"check",theme:"outlined"},d=f,p=e(84089),m=function(g,h){return l.createElement(p.Z,(0,o.Z)({},g,{ref:h,icon:d}))},v=l.forwardRef(m)},46693:function(t,n,e){"use strict";var o=e(97857),l=e.n(o),f=e(13769),d=e.n(f),p=e(16165),m=e(86074),v=["name","width","height"],$=function(c){var w=c.keys().map(function(T){var z=T.replace(/\.\/(.*)\.\w+$/,"$1");return{name:z,value:c(T)}});return w},g=[];try{g=$(e(78131))}catch(P){console.warn(P),g=[]}var h=function(c){var w=c.name,T=c.width,z=c.height,U=d()(c,v),L=g.find(function(k){return k.name===w});return(0,m.jsx)(p.Z,l()({component:function(){return(0,m.jsx)("img",{src:L==null?void 0:L.value,alt:"",width:T,height:z})}},U))};n.Z=h},79495:function(t,n,e){"use strict";e.d(n,{Jf:function(){return C},WH:function(){return L},XH:function(){return z},Zl:function(){return w},aU:function(){return U},fS:function(){return T},jd:function(){return P},ml:function(){return c},nv:function(){return k}});var o=e(5574),l=e.n(o),f=e(15009),d=e.n(f),p=e(99289),m=e.n(p),v=e(85162),$=e(32358),g=e(62435),h=e(6589),P=function(){var _=(0,h.useDispatch)(),b=(0,g.useCallback)(function(){_({type:"settingModel/getUserInfo"})},[_]);(0,g.useEffect)(function(){b()},[b])},c=function(){var _=(0,h.useSelector)(function(b){return b.settingModel.userInfo});return _},w=function(){var _=(0,h.useSelector)(function(b){return b.settingModel.tenantIfo});return _},T=function(){var _=arguments.length>0&&arguments[0]!==void 0?arguments[0]:!0,b=(0,h.useDispatch)(),y=(0,g.useCallback)(function(){b({type:"settingModel/getTenantInfo"})},[b]);return(0,g.useEffect)(function(){_&&y()},[y,_]),y},z=function(){var _=w(),b=(0,g.useMemo)(function(){var y,I=(y=_==null?void 0:_.parser_ids.split(","))!==null&&y!==void 0?y:[];return I.map(function(M){var x=M.split(":");return{value:x[0],label:x[1]}})},[_]);return b},U=function(){var _=(0,h.useDispatch)(),b=(0,g.useCallback)(m()(d()().mark(function y(){var I;return d()().wrap(function(x){for(;;)switch(x.prev=x.next){case 0:return x.next=2,_({type:"loginModel/logout"});case 2:I=x.sent,I===0&&($.Z.removeAll(),h.history.push("/login"));case 4:case"end":return x.stop()}},y)})),[_]);return b},L=function(){var _=(0,h.useDispatch)(),b=(0,g.useCallback)(function(y){return _({type:"settingModel/setting",payload:y})},[_]);return b},k=function(){var _=(0,g.useState)(""),b=l()(_,2),y=b[0],I=b[1],M=(0,g.useState)(!1),x=l()(M,2),F=x[0],W=x[1],G=(0,g.useCallback)(m()(d()().mark(function A(){var B,j;return d()().wrap(function(D){for(;;)switch(D.prev=D.next){case 0:return W(!0),D.next=3,v.Z.getSystemVersion();case 3:B=D.sent,j=B.data,j.retcode===0&&(I(j.data),W(!1));case 6:case"end":return D.stop()}},A)})),[]);return{fetchSystemVersion:G,version:y,loading:F}},C=function(){var _=(0,g.useState)({}),b=l()(_,2),y=b[0],I=b[1],M=(0,g.useState)(!1),x=l()(M,2),F=x[0],W=x[1],G=(0,g.useCallback)(m()(d()().mark(function A(){var B,j;return d()().wrap(function(D){for(;;)switch(D.prev=D.next){case 0:return W(!0),D.next=3,v.Z.getSystemStatus();case 3:B=D.sent,j=B.data,j.retcode===0&&(I(j.data),W(!1));case 6:case"end":return D.stop()}},A)})),[]);return{systemStatus:y,fetchSystemStatus:G,loading:F}}},94544:function(t,n,e){"use strict";e.r(n),e.d(n,{default:function(){return I}});var o=e(9783),l=e.n(o),f=e(46693),d=e(79495),p=e(22850),m=e(75081),v=e(86250),$=e(4393),g=e(40411),h=e(93967),P=e.n(h),c=e(45021),w=e.n(c),T=e(11700),z=e.n(T),U=e(62435),L=e(52043),k={systemInfo:"systemInfo___ufsjg",title:"title___g9daS",text:"text___dsVaC",badge:"badge___lUCbZ",error:"error___DIP2V"},C=e(86074),E=p.Z.Text,_=function(M){return M.green="success",M.red="error",M.yellow="warning",M}(_||{}),b={es:"Elasticsearch",minio:"MinIO Object Storage",redis:"Redis",mysql:"Mysql"},y=function(){var x=(0,d.Jf)(),F=x.systemStatus,W=x.fetchSystemStatus,G=x.loading;return(0,U.useEffect)(function(){W()},[W]),(0,C.jsx)("section",{className:k.systemInfo,children:(0,C.jsx)(m.Z,{spinning:G,children:(0,C.jsx)(v.Z,{gap:16,vertical:!0,children:Object.keys(F).map(function(A){var B=F[A];return(0,C.jsx)($.Z,{type:"inner",title:(0,C.jsxs)(v.Z,{align:"center",gap:10,children:[(0,C.jsx)(f.Z,{name:A,width:26}),(0,C.jsx)("span",{className:k.title,children:b[A]}),(0,C.jsx)(g.Z,{className:k.badge,status:_[B.status]})]}),children:Object.keys(B).filter(function(j){return j!=="status"}).map(function(j){return(0,C.jsxs)(v.Z,{align:"center",gap:16,className:k.text,children:[(0,C.jsxs)("b",{children:[z()(w()(j)),":"]}),(0,C.jsxs)(E,{className:P()(l()({},k.error,j==="error")),children:[(0,L.FH)(B[j]),j==="elapsed"&&" ms"]})]},j)})},A)})})})})},I=y},4393:function(t,n,e){"use strict";e.d(n,{Z:function(){return D}});var o=e(62435),l=e(93967),f=e.n(l),d=e(98423),p=e(53124),m=e(98675),v=e(21687),$=e(92398),g=function(s,u){var a={};for(var r in s)Object.prototype.hasOwnProperty.call(s,r)&&u.indexOf(r)<0&&(a[r]=s[r]);if(s!=null&&typeof Object.getOwnPropertySymbols=="function")for(var i=0,r=Object.getOwnPropertySymbols(s);i<r.length;i++)u.indexOf(r[i])<0&&Object.prototype.propertyIsEnumerable.call(s,r[i])&&(a[r[i]]=s[r[i]]);return a},P=s=>{var{prefixCls:u,className:a,hoverable:r=!0}=s,i=g(s,["prefixCls","className","hoverable"]);const{getPrefixCls:O}=o.useContext(p.E_),R=O("card",u),Z=f()(`${R}-grid`,a,{[`${R}-grid-hoverable`]:r});return o.createElement("div",Object.assign({},i,{className:Z}))},c=e(54548),w=e(14747),T=e(91945),z=e(45503);const U=s=>{const{antCls:u,componentCls:a,headerHeight:r,cardPaddingBase:i,tabsMarginBottom:O}=s;return Object.assign(Object.assign({display:"flex",justifyContent:"center",flexDirection:"column",minHeight:r,marginBottom:-1,padding:`0 ${(0,c.bf)(i)}`,color:s.colorTextHeading,fontWeight:s.fontWeightStrong,fontSize:s.headerFontSize,background:s.headerBg,borderBottom:`${(0,c.bf)(s.lineWidth)} ${s.lineType} ${s.colorBorderSecondary}`,borderRadius:`${(0,c.bf)(s.borderRadiusLG)} ${(0,c.bf)(s.borderRadiusLG)} 0 0`},(0,w.dF)()),{"&-wrapper":{width:"100%",display:"flex",alignItems:"center"},"&-title":Object.assign(Object.assign({display:"inline-block",flex:1},w.vS),{[`
          > ${a}-typography,
          > ${a}-typography-edit-content
        `]:{insetInlineStart:0,marginTop:0,marginBottom:0}}),[`${u}-tabs-top`]:{clear:"both",marginBottom:O,color:s.colorText,fontWeight:"normal",fontSize:s.fontSize,"&-bar":{borderBottom:`${(0,c.bf)(s.lineWidth)} ${s.lineType} ${s.colorBorderSecondary}`}}})},L=s=>{const{cardPaddingBase:u,colorBorderSecondary:a,cardShadow:r,lineWidth:i}=s;return{width:"33.33%",padding:u,border:0,borderRadius:0,boxShadow:`
      ${(0,c.bf)(i)} 0 0 0 ${a},
      0 ${(0,c.bf)(i)} 0 0 ${a},
      ${(0,c.bf)(i)} ${(0,c.bf)(i)} 0 0 ${a},
      ${(0,c.bf)(i)} 0 0 0 ${a} inset,
      0 ${(0,c.bf)(i)} 0 0 ${a} inset;
    `,transition:`all ${s.motionDurationMid}`,"&-hoverable:hover":{position:"relative",zIndex:1,boxShadow:r}}},k=s=>{const{componentCls:u,iconCls:a,actionsLiMargin:r,cardActionsIconSize:i,colorBorderSecondary:O,actionsBg:R}=s;return Object.assign(Object.assign({margin:0,padding:0,listStyle:"none",background:R,borderTop:`${(0,c.bf)(s.lineWidth)} ${s.lineType} ${O}`,display:"flex",borderRadius:`0 0 ${(0,c.bf)(s.borderRadiusLG)} ${(0,c.bf)(s.borderRadiusLG)}`},(0,w.dF)()),{"& > li":{margin:r,color:s.colorTextDescription,textAlign:"center","> span":{position:"relative",display:"block",minWidth:s.calc(s.cardActionsIconSize).mul(2).equal(),fontSize:s.fontSize,lineHeight:s.lineHeight,cursor:"pointer","&:hover":{color:s.colorPrimary,transition:`color ${s.motionDurationMid}`},[`a:not(${u}-btn), > ${a}`]:{display:"inline-block",width:"100%",color:s.colorTextDescription,lineHeight:(0,c.bf)(s.fontHeight),transition:`color ${s.motionDurationMid}`,"&:hover":{color:s.colorPrimary}},[`> ${a}`]:{fontSize:i,lineHeight:(0,c.bf)(s.calc(i).mul(s.lineHeight).equal())}},"&:not(:last-child)":{borderInlineEnd:`${(0,c.bf)(s.lineWidth)} ${s.lineType} ${O}`}}})},C=s=>Object.assign(Object.assign({margin:`${(0,c.bf)(s.calc(s.marginXXS).mul(-1).equal())} 0`,display:"flex"},(0,w.dF)()),{"&-avatar":{paddingInlineEnd:s.padding},"&-detail":{overflow:"hidden",flex:1,"> div:not(:last-child)":{marginBottom:s.marginXS}},"&-title":Object.assign({color:s.colorTextHeading,fontWeight:s.fontWeightStrong,fontSize:s.fontSizeLG},w.vS),"&-description":{color:s.colorTextDescription}}),E=s=>{const{componentCls:u,cardPaddingBase:a,colorFillAlter:r}=s;return{[`${u}-head`]:{padding:`0 ${(0,c.bf)(a)}`,background:r,"&-title":{fontSize:s.fontSize}},[`${u}-body`]:{padding:`${(0,c.bf)(s.padding)} ${(0,c.bf)(a)}`}}},_=s=>{const{componentCls:u}=s;return{overflow:"hidden",[`${u}-body`]:{userSelect:"none"}}},b=s=>{const{antCls:u,componentCls:a,cardShadow:r,cardHeadPadding:i,colorBorderSecondary:O,boxShadowTertiary:R,cardPaddingBase:Z,extraColor:N}=s;return{[a]:Object.assign(Object.assign({},(0,w.Wf)(s)),{position:"relative",background:s.colorBgContainer,borderRadius:s.borderRadiusLG,[`&:not(${a}-bordered)`]:{boxShadow:R},[`${a}-head`]:U(s),[`${a}-extra`]:{marginInlineStart:"auto",color:N,fontWeight:"normal",fontSize:s.fontSize},[`${a}-body`]:Object.assign({padding:Z,borderRadius:` 0 0 ${(0,c.bf)(s.borderRadiusLG)} ${(0,c.bf)(s.borderRadiusLG)}`},(0,w.dF)()),[`${a}-grid`]:L(s),[`${a}-cover`]:{"> *":{display:"block",width:"100%"},[`img, img + ${u}-image-mask`]:{borderRadius:`${(0,c.bf)(s.borderRadiusLG)} ${(0,c.bf)(s.borderRadiusLG)} 0 0`}},[`${a}-actions`]:k(s),[`${a}-meta`]:C(s)}),[`${a}-bordered`]:{border:`${(0,c.bf)(s.lineWidth)} ${s.lineType} ${O}`,[`${a}-cover`]:{marginTop:-1,marginInlineStart:-1,marginInlineEnd:-1}},[`${a}-hoverable`]:{cursor:"pointer",transition:`box-shadow ${s.motionDurationMid}, border-color ${s.motionDurationMid}`,"&:hover":{borderColor:"transparent",boxShadow:r}},[`${a}-contain-grid`]:{borderRadius:`${(0,c.bf)(s.borderRadiusLG)} ${(0,c.bf)(s.borderRadiusLG)} 0 0 `,[`${a}-body`]:{display:"flex",flexWrap:"wrap"},[`&:not(${a}-loading) ${a}-body`]:{marginBlockStart:s.calc(s.lineWidth).mul(-1).equal(),marginInlineStart:s.calc(s.lineWidth).mul(-1).equal(),padding:0}},[`${a}-contain-tabs`]:{[`> ${a}-head`]:{minHeight:0,[`${a}-head-title, ${a}-extra`]:{paddingTop:i}}},[`${a}-type-inner`]:E(s),[`${a}-loading`]:_(s),[`${a}-rtl`]:{direction:"rtl"}}},y=s=>{const{componentCls:u,cardPaddingSM:a,headerHeightSM:r,headerFontSizeSM:i}=s;return{[`${u}-small`]:{[`> ${u}-head`]:{minHeight:r,padding:`0 ${(0,c.bf)(a)}`,fontSize:i,[`> ${u}-head-wrapper`]:{[`> ${u}-extra`]:{fontSize:s.fontSize}}},[`> ${u}-body`]:{padding:a}},[`${u}-small${u}-contain-tabs`]:{[`> ${u}-head`]:{[`${u}-head-title, ${u}-extra`]:{paddingTop:0,display:"flex",alignItems:"center"}}}}},I=s=>({headerBg:"transparent",headerFontSize:s.fontSizeLG,headerFontSizeSM:s.fontSize,headerHeight:s.fontSizeLG*s.lineHeightLG+s.padding*2,headerHeightSM:s.fontSize*s.lineHeight+s.paddingXS*2,actionsBg:s.colorBgContainer,actionsLiMargin:`${s.paddingSM}px 0`,tabsMarginBottom:-s.padding-s.lineWidth,extraColor:s.colorText});var M=(0,T.I$)("Card",s=>{const u=(0,z.TS)(s,{cardShadow:s.boxShadowCard,cardHeadPadding:s.padding,cardPaddingBase:s.paddingLG,cardActionsIconSize:s.fontSize,cardPaddingSM:12});return[b(u),y(u)]},I),x=function(s,u){var a={};for(var r in s)Object.prototype.hasOwnProperty.call(s,r)&&u.indexOf(r)<0&&(a[r]=s[r]);if(s!=null&&typeof Object.getOwnPropertySymbols=="function")for(var i=0,r=Object.getOwnPropertySymbols(s);i<r.length;i++)u.indexOf(r[i])<0&&Object.prototype.propertyIsEnumerable.call(s,r[i])&&(a[r[i]]=s[r[i]]);return a};const F=s=>{const{prefixCls:u,actions:a=[]}=s;return o.createElement("ul",{className:`${u}-actions`},a.map((r,i)=>{const O=`action-${i}`;return o.createElement("li",{style:{width:`${100/a.length}%`},key:O},o.createElement("span",null,r))}))};var G=o.forwardRef((s,u)=>{const{prefixCls:a,className:r,rootClassName:i,style:O,extra:R,headStyle:Z={},bodyStyle:N={},title:X,loading:Y,bordered:q=!0,size:ee,type:te,cover:re,actions:ae,tabList:se,children:ne,activeTabKey:oe,defaultActiveTabKey:de,tabBarExtraContent:pe,hoverable:le,tabProps:_e={}}=s,fe=x(s,["prefixCls","className","rootClassName","style","extra","headStyle","bodyStyle","title","loading","bordered","size","type","cover","actions","tabList","children","activeTabKey","defaultActiveTabKey","tabBarExtraContent","hoverable","tabProps"]),{getPrefixCls:ge,direction:ve,card:V}=o.useContext(p.E_),be=K=>{var H;(H=s.onTabChange)===null||H===void 0||H.call(s,K)},me=o.useMemo(()=>{let K=!1;return o.Children.forEach(ne,H=>{H&&H.type&&H.type===P&&(K=!0)}),K},[ne]),S=ge("card",a),[xe,he,we]=M(S),ye=o.createElement(v.Z,{loading:!0,active:!0,paragraph:{rows:4},title:!1},ne),ie=oe!==void 0,Se=Object.assign(Object.assign({},_e),{[ie?"activeKey":"defaultActiveKey"]:ie?oe:de,tabBarExtraContent:pe});let ce;const Q=(0,m.Z)(ee),$e=!Q||Q==="default"?"large":Q,ue=se?o.createElement($.Z,Object.assign({size:$e},Se,{className:`${S}-head-tabs`,onChange:be,items:se.map(K=>{var{tab:H}=K,Pe=x(K,["tab"]);return Object.assign({label:H},Pe)})})):null;(X||R||ue)&&(ce=o.createElement("div",{className:`${S}-head`,style:Z},o.createElement("div",{className:`${S}-head-wrapper`},X&&o.createElement("div",{className:`${S}-head-title`},X),R&&o.createElement("div",{className:`${S}-extra`},R)),ue));const Ce=re?o.createElement("div",{className:`${S}-cover`},re):null,je=o.createElement("div",{className:`${S}-body`,style:N},Y?ye:ne),Oe=ae&&ae.length?o.createElement(F,{prefixCls:S,actions:ae}):null,ke=(0,d.Z)(fe,["onTabChange"]),Ee=f()(S,V==null?void 0:V.className,{[`${S}-loading`]:Y,[`${S}-bordered`]:q,[`${S}-hoverable`]:le,[`${S}-contain-grid`]:me,[`${S}-contain-tabs`]:se&&se.length,[`${S}-${Q}`]:Q,[`${S}-type-${te}`]:!!te,[`${S}-rtl`]:ve==="rtl"},r,i,he,we),Me=Object.assign(Object.assign({},V==null?void 0:V.style),O);return xe(o.createElement("div",Object.assign({ref:u},ke,{className:Ee,style:Me}),ce,Ce,je,Oe))}),A=function(s,u){var a={};for(var r in s)Object.prototype.hasOwnProperty.call(s,r)&&u.indexOf(r)<0&&(a[r]=s[r]);if(s!=null&&typeof Object.getOwnPropertySymbols=="function")for(var i=0,r=Object.getOwnPropertySymbols(s);i<r.length;i++)u.indexOf(r[i])<0&&Object.prototype.propertyIsEnumerable.call(s,r[i])&&(a[r[i]]=s[r[i]]);return a},j=s=>{const{prefixCls:u,className:a,avatar:r,title:i,description:O}=s,R=A(s,["prefixCls","className","avatar","title","description"]),{getPrefixCls:Z}=o.useContext(p.E_),N=Z("card",u),X=f()(`${N}-meta`,a),Y=r?o.createElement("div",{className:`${N}-meta-avatar`},r):null,q=i?o.createElement("div",{className:`${N}-meta-title`},i):null,ee=O?o.createElement("div",{className:`${N}-meta-description`},O):null,te=q||ee?o.createElement("div",{className:`${N}-meta-detail`},q,ee):null;return o.createElement("div",Object.assign({},R,{className:X}),Y,te)};const J=G;J.Grid=P,J.Meta=j;var D=J},44286:function(t){function n(e){return e.split("")}t.exports=n},40180:function(t,n,e){var o=e(14259);function l(f,d,p){var m=f.length;return p=p===void 0?m:p,!d&&p>=m?f:o(f,d,p)}t.exports=l},98805:function(t,n,e){var o=e(40180),l=e(62689),f=e(83140),d=e(79833);function p(m){return function(v){v=d(v);var $=l(v)?f(v):void 0,g=$?$[0]:v.charAt(0),h=$?o($,1).join(""):v.slice(1);return g[m]()+h}}t.exports=p},62689:function(t){var n="\\ud800-\\udfff",e="\\u0300-\\u036f",o="\\ufe20-\\ufe2f",l="\\u20d0-\\u20ff",f=e+o+l,d="\\ufe0e\\ufe0f",p="\\u200d",m=RegExp("["+p+n+f+d+"]");function v($){return m.test($)}t.exports=v},83140:function(t,n,e){var o=e(44286),l=e(62689),f=e(676);function d(p){return l(p)?f(p):o(p)}t.exports=d},676:function(t){var n="\\ud800-\\udfff",e="\\u0300-\\u036f",o="\\ufe20-\\ufe2f",l="\\u20d0-\\u20ff",f=e+o+l,d="\\ufe0e\\ufe0f",p="["+n+"]",m="["+f+"]",v="\\ud83c[\\udffb-\\udfff]",$="(?:"+m+"|"+v+")",g="[^"+n+"]",h="(?:\\ud83c[\\udde6-\\uddff]){2}",P="[\\ud800-\\udbff][\\udc00-\\udfff]",c="\\u200d",w=$+"?",T="["+d+"]?",z="(?:"+c+"(?:"+[g,h,P].join("|")+")"+T+w+")*",U=T+w+z,L="(?:"+[g+m+"?",m,h,P,p].join("|")+")",k=RegExp(v+"(?="+v+")|"+L+U,"g");function C(E){return E.match(k)||[]}t.exports=C},45021:function(t,n,e){var o=e(35393),l=o(function(f,d,p){return f+(p?" ":"")+d.toLowerCase()});t.exports=l},11700:function(t,n,e){var o=e(98805),l=o("toUpperCase");t.exports=l},78131:function(t,n,e){var o={"./assistant.svg":4159,"./cancel.svg":8156,"./chat-app-cube.svg":34339,"./chat-configuration-atom.svg":57950,"./chat-star.svg":49570,"./chunk-method/book-01.svg":38726,"./chunk-method/book-02.svg":21357,"./chunk-method/book-03.svg":88093,"./chunk-method/book-04.svg":58355,"./chunk-method/chunk-empty.svg":16035,"./chunk-method/law-01.svg":54530,"./chunk-method/law-02.svg":75787,"./chunk-method/manual-01.svg":64179,"./chunk-method/manual-02.svg":40547,"./chunk-method/manual-03.svg":1437,"./chunk-method/manual-04.svg":81486,"./chunk-method/media-01.svg":33952,"./chunk-method/media-02.svg":93802,"./chunk-method/naive-01.svg":88488,"./chunk-method/naive-02.svg":74795,"./chunk-method/one-01.svg":88533,"./chunk-method/one-02.svg":99199,"./chunk-method/one-03.svg":23192,"./chunk-method/one-04.svg":72906,"./chunk-method/paper-01.svg":64018,"./chunk-method/paper-02.svg":77250,"./chunk-method/presentation-01.svg":87162,"./chunk-method/presentation-02.svg":59640,"./chunk-method/qa-01.svg":39420,"./chunk-method/qa-02.svg":55086,"./chunk-method/resume-01.svg":76604,"./chunk-method/resume-02.svg":96131,"./chunk-method/table-01.svg":42743,"./chunk-method/table-02.svg":84817,"./delete.svg":71483,"./disable.svg":84377,"./enable.svg":8207,"./es.svg":96492,"./file-icon/aep.svg":83889,"./file-icon/ai.svg":28897,"./file-icon/avi.svg":27635,"./file-icon/css.svg":52229,"./file-icon/csv.svg":50747,"./file-icon/dmg.svg":71691,"./file-icon/doc.svg":35354,"./file-icon/docx.svg":69139,"./file-icon/eps.svg":43279,"./file-icon/exe.svg":46462,"./file-icon/fig.svg":33179,"./file-icon/folder.svg":54168,"./file-icon/gif.svg":73845,"./file-icon/html.svg":53682,"./file-icon/indd.svg":26763,"./file-icon/java.svg":13916,"./file-icon/jpeg.svg":19525,"./file-icon/jpg.svg":24674,"./file-icon/js.svg":38470,"./file-icon/json.svg":56373,"./file-icon/mkv.svg":49197,"./file-icon/mp3.svg":44538,"./file-icon/mp4.svg":52038,"./file-icon/mpeg.svg":77747,"./file-icon/pdf.svg":82938,"./file-icon/png.svg":71915,"./file-icon/ppt.svg":85173,"./file-icon/pptx.svg":62133,"./file-icon/psd.svg":30182,"./file-icon/rss.svg":38428,"./file-icon/sql.svg":78501,"./file-icon/svg.svg":72515,"./file-icon/tiff.svg":4502,"./file-icon/txt.svg":55773,"./file-icon/wav.svg":63645,"./file-icon/webp.svg":87260,"./file-icon/xls.svg":5343,"./file-icon/xlsx.svg":41475,"./file-icon/xml.svg":38476,"./file-management.svg":39121,"./knowledge-base.svg":87055,"./knowledge-configration.svg":96450,"./knowledge-dataset.svg":9385,"./knowledge-testing.svg":65376,"./llm/baichuan.svg":6507,"./llm/deepseek.svg":23476,"./llm/github.svg":29034,"./llm/google.svg":93926,"./llm/jina.svg":19765,"./llm/moonshot.svg":98184,"./llm/ollama.svg":59488,"./llm/openai.svg":81459,"./llm/tongyi.svg":74296,"./llm/volc_engine.svg":25210,"./llm/wenxin.svg":34981,"./llm/xinference.svg":76577,"./llm/zhipu.svg":19568,"./login-avatars.svg":68683,"./login-background.svg":89624,"./login-star.svg":31130,"./logout.svg":67487,"./minio.svg":32382,"./model-providers.svg":70410,"./moon.svg":22350,"./more-model.svg":88458,"./more.svg":1333,"./mysql.svg":37124,"./navigation-pointer.svg":1979,"./password.svg":33530,"./profile.svg":1769,"./redis.svg":43867,"./refresh.svg":96198,"./run.svg":91449,"./select-files-end.svg":14387,"./select-files-start.svg":16734,"./selected-files-collapse.svg":40975,"./team.svg":52381,"./translation.svg":40724};function l(d){var p=f(d);return e(p)}function f(d){if(!e.o(o,d)){var p=new Error("Cannot find module '"+d+"'");throw p.code="MODULE_NOT_FOUND",p}return o[d]}l.keys=function(){return Object.keys(o)},l.resolve=f,t.exports=l,l.id=78131},4159:function(t,n,e){"use strict";t.exports=e.p+"static/assistant.66de9b57.svg"},8156:function(t,n,e){"use strict";t.exports=e.p+"static/cancel.4450bdfb.svg"},34339:function(t,n,e){"use strict";t.exports=e.p+"static/chat-app-cube.0d35727d.svg"},57950:function(t,n,e){"use strict";t.exports=e.p+"static/chat-configuration-atom.94c02c7a.svg"},49570:function(t,n,e){"use strict";t.exports=e.p+"static/chat-star.7acfee68.svg"},38726:function(t,n,e){"use strict";t.exports=e.p+"static/book-01.6e2a8a37.svg"},21357:function(t,n,e){"use strict";t.exports=e.p+"static/book-02.4dd45490.svg"},88093:function(t,n,e){"use strict";t.exports=e.p+"static/book-03.bd76b691.svg"},58355:function(t,n,e){"use strict";t.exports=e.p+"static/book-04.594d0d1a.svg"},16035:function(t,n,e){"use strict";t.exports=e.p+"static/chunk-empty.2b83673a.svg"},54530:function(t,n,e){"use strict";t.exports=e.p+"static/law-01.7070b7b3.svg"},75787:function(t,n,e){"use strict";t.exports=e.p+"static/law-02.cae944e5.svg"},64179:function(t,n,e){"use strict";t.exports=e.p+"static/manual-01.e3bb11d2.svg"},40547:function(t,n,e){"use strict";t.exports=e.p+"static/manual-02.1a214f22.svg"},1437:function(t,n,e){"use strict";t.exports=e.p+"static/manual-03.530928ca.svg"},81486:function(t,n,e){"use strict";t.exports=e.p+"static/manual-04.63d43265.svg"},33952:function(t,n,e){"use strict";t.exports=e.p+"static/media-01.086483b6.svg"},93802:function(t,n,e){"use strict";t.exports=e.p+"static/media-02.d4c36e3e.svg"},88488:function(t,n,e){"use strict";t.exports=e.p+"static/naive-01.f57569b7.svg"},74795:function(t,n,e){"use strict";t.exports=e.p+"static/naive-02.3fe3610b.svg"},88533:function(t,n,e){"use strict";t.exports=e.p+"static/one-01.5a1d6960.svg"},99199:function(t,n,e){"use strict";t.exports=e.p+"static/one-02.0adb16f8.svg"},23192:function(t,n,e){"use strict";t.exports=e.p+"static/one-03.466dec02.svg"},72906:function(t,n,e){"use strict";t.exports=e.p+"static/one-04.3b10ee6d.svg"},64018:function(t,n,e){"use strict";t.exports=e.p+"static/paper-01.e0019dcd.svg"},77250:function(t,n,e){"use strict";t.exports=e.p+"static/paper-02.cf73a211.svg"},87162:function(t,n,e){"use strict";t.exports=e.p+"static/presentation-01.06115027.svg"},59640:function(t,n,e){"use strict";t.exports=e.p+"static/presentation-02.14d98352.svg"},39420:function(t,n,e){"use strict";t.exports=e.p+"static/qa-01.ce8702ee.svg"},55086:function(t,n,e){"use strict";t.exports=e.p+"static/qa-02.c181fcd6.svg"},76604:function(t,n,e){"use strict";t.exports=e.p+"static/resume-01.75f1c93f.svg"},96131:function(t,n,e){"use strict";t.exports=e.p+"static/resume-02.9c115ed1.svg"},42743:function(t,n,e){"use strict";t.exports=e.p+"static/table-01.ec5d8a4d.svg"},84817:function(t,n,e){"use strict";t.exports=e.p+"static/table-02.e4d2487c.svg"},71483:function(t,n,e){"use strict";t.exports=e.p+"static/delete.b9891386.svg"},84377:function(t,n,e){"use strict";t.exports=e.p+"static/disable.55c8b50f.svg"},8207:function(t,n,e){"use strict";t.exports=e.p+"static/enable.1b0d90c7.svg"},96492:function(t,n,e){"use strict";t.exports=e.p+"static/es.d9969bbd.svg"},83889:function(t,n,e){"use strict";t.exports=e.p+"static/aep.90be9439.svg"},28897:function(t,n,e){"use strict";t.exports=e.p+"static/ai.10138190.svg"},27635:function(t,n,e){"use strict";t.exports=e.p+"static/avi.02b94047.svg"},52229:function(t,n,e){"use strict";t.exports=e.p+"static/css.8f0ad31d.svg"},50747:function(t,n,e){"use strict";t.exports=e.p+"static/csv.9a256b45.svg"},71691:function(t,n,e){"use strict";t.exports=e.p+"static/dmg.57f9c02c.svg"},35354:function(t,n,e){"use strict";t.exports=e.p+"static/doc.d8918cc4.svg"},69139:function(t,n,e){"use strict";t.exports=e.p+"static/docx.38d543b1.svg"},43279:function(t,n,e){"use strict";t.exports=e.p+"static/eps.3f104d7d.svg"},46462:function(t,n,e){"use strict";t.exports=e.p+"static/exe.7c93a166.svg"},33179:function(t,n,e){"use strict";t.exports=e.p+"static/fig.38a31555.svg"},54168:function(t,n,e){"use strict";t.exports=e.p+"static/folder.84bcc794.svg"},73845:function(t,n,e){"use strict";t.exports=e.p+"static/gif.06cc115a.svg"},53682:function(t,n,e){"use strict";t.exports=e.p+"static/html.240ba9a0.svg"},26763:function(t,n,e){"use strict";t.exports=e.p+"static/indd.22c71a5d.svg"},13916:function(t,n,e){"use strict";t.exports=e.p+"static/java.bc10ed5b.svg"},19525:function(t,n,e){"use strict";t.exports=e.p+"static/jpeg.b4ff21a3.svg"},24674:function(t,n,e){"use strict";t.exports=e.p+"static/jpg.166b6e5d.svg"},38470:function(t,n,e){"use strict";t.exports=e.p+"static/js.acbe367d.svg"},56373:function(t,n,e){"use strict";t.exports=e.p+"static/json.79aa2433.svg"},49197:function(t,n,e){"use strict";t.exports=e.p+"static/mkv.ed2674b7.svg"},44538:function(t,n,e){"use strict";t.exports=e.p+"static/mp3.773e22e6.svg"},52038:function(t,n,e){"use strict";t.exports=e.p+"static/mp4.4b08cc18.svg"},77747:function(t,n,e){"use strict";t.exports=e.p+"static/mpeg.ca6e2469.svg"},82938:function(t,n,e){"use strict";t.exports=e.p+"static/pdf.44344347.svg"},71915:function(t,n,e){"use strict";t.exports=e.p+"static/png.b6606e2b.svg"},85173:function(t,n,e){"use strict";t.exports=e.p+"static/ppt.1cb25ad9.svg"},62133:function(t,n,e){"use strict";t.exports=e.p+"static/pptx.b108e970.svg"},30182:function(t,n,e){"use strict";t.exports=e.p+"static/psd.1d66388a.svg"},38428:function(t,n,e){"use strict";t.exports=e.p+"static/rss.f27341d3.svg"},78501:function(t,n,e){"use strict";t.exports=e.p+"static/sql.f90e0e1d.svg"},72515:function(t,n,e){"use strict";t.exports=e.p+"static/svg.a95ef072.svg"},4502:function(t,n,e){"use strict";t.exports=e.p+"static/tiff.4719932c.svg"},55773:function(t,n,e){"use strict";t.exports=e.p+"static/txt.ef59d3d8.svg"},63645:function(t,n,e){"use strict";t.exports=e.p+"static/wav.fa097b95.svg"},87260:function(t,n,e){"use strict";t.exports=e.p+"static/webp.ef46db37.svg"},5343:function(t,n,e){"use strict";t.exports=e.p+"static/xls.2ba7600c.svg"},41475:function(t,n,e){"use strict";t.exports=e.p+"static/xlsx.746a1908.svg"},38476:function(t,n,e){"use strict";t.exports=e.p+"static/xml.b90e705e.svg"},39121:function(t,n,e){"use strict";t.exports=e.p+"static/file-management.b76487d8.svg"},87055:function(t,n,e){"use strict";t.exports=e.p+"static/knowledge-base.1c6120ee.svg"},96450:function(t,n,e){"use strict";t.exports=e.p+"static/knowledge-configration.852175c9.svg"},9385:function(t,n,e){"use strict";t.exports=e.p+"static/knowledge-dataset.722b6fe7.svg"},65376:function(t,n,e){"use strict";t.exports=e.p+"static/knowledge-testing.bde5e258.svg"},6507:function(t,n,e){"use strict";t.exports=e.p+"static/baichuan.e3f694dc.svg"},23476:function(t,n,e){"use strict";t.exports=e.p+"static/deepseek.f974cd8d.svg"},29034:function(t,n,e){"use strict";t.exports=e.p+"static/github.6dbd4f80.svg"},93926:function(t,n,e){"use strict";t.exports=e.p+"static/google.36013c77.svg"},19765:function(t,n,e){"use strict";t.exports=e.p+"static/jina.1aa22682.svg"},98184:function(t,n,e){"use strict";t.exports=e.p+"static/moonshot.3e35964c.svg"},59488:function(t,n,e){"use strict";t.exports=e.p+"static/ollama.2677a93a.svg"},81459:function(t,n,e){"use strict";t.exports=e.p+"static/openai.6242ead4.svg"},74296:function(t,n,e){"use strict";t.exports=e.p+"static/tongyi.8c1b0f0d.svg"},25210:function(t,n,e){"use strict";t.exports=e.p+"static/volc_engine.c93921c9.svg"},34981:function(t,n,e){"use strict";t.exports=e.p+"static/wenxin.c029f1ef.svg"},76577:function(t,n,e){"use strict";t.exports=e.p+"static/xinference.4bf1c9ad.svg"},19568:function(t,n,e){"use strict";t.exports=e.p+"static/zhipu.53c4367a.svg"},68683:function(t,n,e){"use strict";t.exports=e.p+"static/login-avatars.2cd59ec8.svg"},89624:function(t,n,e){"use strict";t.exports=e.p+"static/login-background.53821b0f.svg"},31130:function(t,n,e){"use strict";t.exports=e.p+"static/login-star.a1de9742.svg"},67487:function(t,n,e){"use strict";t.exports=e.p+"static/logout.acbe92a8.svg"},32382:function(t,n,e){"use strict";t.exports=e.p+"static/minio.2aa38883.svg"},70410:function(t,n,e){"use strict";t.exports=e.p+"static/model-providers.43583ddb.svg"},22350:function(t,n,e){"use strict";t.exports=e.p+"static/moon.15e7f056.svg"},88458:function(t,n,e){"use strict";t.exports=e.p+"static/more-model.7a1d39d8.svg"},1333:function(t,n,e){"use strict";t.exports=e.p+"static/more.4f8a95a4.svg"},37124:function(t,n,e){"use strict";t.exports=e.p+"static/mysql.cb3593ae.svg"},1979:function(t,n,e){"use strict";t.exports=e.p+"static/navigation-pointer.a22fd9df.svg"},33530:function(t,n,e){"use strict";t.exports=e.p+"static/password.668a7506.svg"},1769:function(t,n,e){"use strict";t.exports=e.p+"static/profile.67855e30.svg"},43867:function(t,n,e){"use strict";t.exports=e.p+"static/redis.c34741f7.svg"},96198:function(t,n,e){"use strict";t.exports=e.p+"static/refresh.8782839f.svg"},91449:function(t,n,e){"use strict";t.exports=e.p+"static/run.23b47028.svg"},14387:function(t,n,e){"use strict";t.exports=e.p+"static/select-files-end.c089e39f.svg"},16734:function(t,n,e){"use strict";t.exports=e.p+"static/select-files-start.fc6d0609.svg"},40975:function(t,n,e){"use strict";t.exports=e.p+"static/selected-files-collapse.9b76e30e.svg"},52381:function(t,n,e){"use strict";t.exports=e.p+"static/team.40ab5e28.svg"},40724:function(t,n,e){"use strict";t.exports=e.p+"static/translation.1d6d8a4a.svg"}}]);

//# sourceMappingURL=p__user-setting__setting-system__index.d9898a5f.async.js.map