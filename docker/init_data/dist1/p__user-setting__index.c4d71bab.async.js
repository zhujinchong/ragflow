(self.webpackChunk=self.webpackChunk||[]).push([[5410],{70101:function(h,d,e){"use strict";e.d(d,{Z:function(){return c}});var s=e(87462),f=e(62435),E={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M692.8 412.7l.2-.2-34.6-44.3a7.97 7.97 0 00-11.2-1.4l-50.4 39.3-70.5-90.1a7.97 7.97 0 00-11.2-1.4l-37.9 29.7a7.97 7.97 0 00-1.4 11.2l70.5 90.2-.2.1 34.6 44.3c2.7 3.5 7.7 4.1 11.2 1.4l50.4-39.3 64.1 82c2.7 3.5 7.7 4.1 11.2 1.4l37.9-29.6c3.5-2.7 4.1-7.7 1.4-11.2l-64.1-82.1zM608 112c-167.9 0-304 136.1-304 304 0 70.3 23.9 135 63.9 186.5L114.3 856.1a8.03 8.03 0 000 11.3l42.3 42.3c3.1 3.1 8.2 3.1 11.3 0l253.6-253.6C473 696.1 537.7 720 608 720c167.9 0 304-136.1 304-304S775.9 112 608 112zm161.2 465.2C726.2 620.3 668.9 644 608 644s-118.2-23.7-161.2-66.8C403.7 534.2 380 476.9 380 416s23.7-118.2 66.8-161.2c43-43.1 100.3-66.8 161.2-66.8s118.2 23.7 161.2 66.8c43.1 43 66.8 100.3 66.8 161.2s-23.7 118.2-66.8 161.2z"}}]},name:"monitor",theme:"outlined"},m=E,g=e(84089),v=function(n,P){return f.createElement(g.Z,(0,s.Z)({},n,{ref:P,icon:m}))},c=f.forwardRef(v)},96330:function(h,d,e){"use strict";e.d(d,{Rx:function(){return E},S8:function(){return v},Vr:function(){return c},cG:function(){return m},ld:function(){return D},oQ:function(){return g}});var s=e(9783),f=e.n(s),E=function(n){return n.Dataset="dataset",n.Testing="testing",n.Configuration="configuration",n}({}),m=function(n){return n.UNSTART="0",n.RUNNING="1",n.CANCEL="2",n.DONE="3",n.FAIL="4",n}({}),g=function(n){return n.Improvise="Improvise",n.Precise="Precise",n.Balance="Balance",n}({}),v=f()(f()(f()({},g.Improvise,{temperature:.9,top_p:.9,frequency_penalty:.2,presence_penalty:.4,max_tokens:512}),g.Precise,{temperature:.1,top_p:.3,frequency_penalty:.7,presence_penalty:.4,max_tokens:512}),g.Balance,{temperature:.5,top_p:.5,frequency_penalty:.7,presence_penalty:.4,max_tokens:512}),c=function(n){return n.Embedding="embedding",n.Chat="chat",n.Image2text="image2text",n.Speech2text="speech2text",n.Rerank="rerank",n}({}),D=function(n){return n.DocumentId="doc_id",n.KnowledgeId="id",n}({})},2039:function(h,d,e){"use strict";e.d(d,{I3:function(){return j},pG:function(){return T},qM:function(){return I}});var s=e(15009),f=e.n(s),E=e(99289),m=e.n(E),g=e(5574),v=e.n(g),c=e(21640),D=e(3321),n=e(18446),P=e.n(n),y=e(62435),S=e(67421),b=e(86074),T=function(){var t=(0,y.useState)(!1),i=v()(t,2),o=i[0],a=i[1],A=function(){a(!0)},p=function(){a(!1)},r=function(){a(!o)};return{visible:o,showModal:A,hideModal:p,switchVisible:r}},U=function(t,i){var o=useRef(),a=function(){};isEqual(i,o.current)||(a=t(),o.current=i),useEffect(function(){return function(){a&&a()}},[])};function M(C){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},i=useRef(),o=useState(!1),a=_slicedToArray(o,2),A=a[0],p=a[1],r=useState(),l=_slicedToArray(r,2),_=l[0],u=l[1],O=t.onCompleted,x=t.onError;return useEffect(function(){p(!0);var W=function(){var R=_asyncToGenerator(_regeneratorRuntime().mark(function w(){return _regeneratorRuntime().wrap(function(B){for(;;)switch(B.prev=B.next){case 0:return B.prev=0,B.next=3,e(86635)(C);case 3:i.current=B.sent.ReactComponent,O==null||O(C,i.current),B.next=11;break;case 7:B.prev=7,B.t0=B.catch(0),x==null||x(B.t0),u(B.t0);case 11:return B.prev=11,p(!1),B.finish(11);case 14:case"end":return B.stop()}},w,null,[[0,7,11,14]])}));return function(){return R.apply(this,arguments)}}();W()},[C,O,x]),{error:_,loading:A,SvgIcon:i.current}}var j=function(){var t=D.Z.useApp(),i=t.modal,o=(0,S.$G)(),a=o.t,A=(0,y.useCallback)(function(p){var r=p.onOk,l=p.onCancel;return new Promise(function(_,u){i.confirm({title:a("common.deleteModalTitle"),icon:(0,b.jsx)(c.Z,{}),okText:a("common.ok"),okType:"danger",cancelText:a("common.cancel"),onOk:function(){return m()(f()().mark(function x(){var W;return f()().wrap(function(w){for(;;)switch(w.prev=w.next){case 0:return w.prev=0,w.next=3,r==null?void 0:r();case 3:W=w.sent,_(W),console.info(W),w.next=11;break;case 8:w.prev=8,w.t0=w.catch(0),u(w.t0);case 11:case"end":return w.stop()}},x,null,[[0,8]])}))()},onCancel:function(){l==null||l()}})})},[a,i]);return A},I=function(t){return(0,S.$G)("translation",{keyPrefix:t})},L=function(){return useTranslation("translation",{keyPrefix:"common"})}},65534:function(h,d,e){"use strict";e.d(d,{HK:function(){return n},Uu:function(){return D},nT:function(){return S},pE:function(){return P},wy:function(){return y}});var s=e(5574),f=e.n(s),E=e(96330),m=e(62435),g=e(6589),v=function(b){return b.Second="2",b.Third="3",b}({}),c=function(T){var U=(0,g.useLocation)(),M=U.pathname,j=M.split("/");return j[T]||""},D=function(){return c(v.Second)},n=function(){return c(v.Third)},P=function(){var T=(0,g.useSearchParams)(),U=f()(T,1),M=U[0];return{documentId:M.get(E.ld.DocumentId)||"",knowledgeId:M.get(E.ld.KnowledgeId)||""}},y=function(){var T=(0,g.useNavigate)();return(0,m.useCallback)(function(U){T(U,{state:{from:U}})},[T])},S=function(){var T=(0,g.useNavigate)(),U=P(),M=U.knowledgeId;return(0,m.useCallback)(function(){T("/knowledge/".concat(E.Rx.Dataset,"?id=").concat(M))},[M,T])}},79495:function(h,d,e){"use strict";e.d(d,{Jf:function(){return L},WH:function(){return j},XH:function(){return U},Zl:function(){return b},aU:function(){return M},fS:function(){return T},jd:function(){return y},ml:function(){return S},nv:function(){return I}});var s=e(5574),f=e.n(s),E=e(15009),m=e.n(E),g=e(99289),v=e.n(g),c=e(85162),D=e(32358),n=e(62435),P=e(6589),y=function(){var t=(0,P.useDispatch)(),i=(0,n.useCallback)(function(){t({type:"settingModel/getUserInfo"})},[t]);(0,n.useEffect)(function(){i()},[i])},S=function(){var t=(0,P.useSelector)(function(i){return i.settingModel.userInfo});return t},b=function(){var t=(0,P.useSelector)(function(i){return i.settingModel.tenantIfo});return t},T=function(){var t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:!0,i=(0,P.useDispatch)(),o=(0,n.useCallback)(function(){i({type:"settingModel/getTenantInfo"})},[i]);return(0,n.useEffect)(function(){t&&o()},[o,t]),o},U=function(){var t=b(),i=(0,n.useMemo)(function(){var o,a=(o=t==null?void 0:t.parser_ids.split(","))!==null&&o!==void 0?o:[];return a.map(function(A){var p=A.split(":");return{value:p[0],label:p[1]}})},[t]);return i},M=function(){var t=(0,P.useDispatch)(),i=(0,n.useCallback)(v()(m()().mark(function o(){var a;return m()().wrap(function(p){for(;;)switch(p.prev=p.next){case 0:return p.next=2,t({type:"loginModel/logout"});case 2:a=p.sent,a===0&&(D.Z.removeAll(),P.history.push("/login"));case 4:case"end":return p.stop()}},o)})),[t]);return i},j=function(){var t=(0,P.useDispatch)(),i=(0,n.useCallback)(function(o){return t({type:"settingModel/setting",payload:o})},[t]);return i},I=function(){var t=(0,n.useState)(""),i=f()(t,2),o=i[0],a=i[1],A=(0,n.useState)(!1),p=f()(A,2),r=p[0],l=p[1],_=(0,n.useCallback)(v()(m()().mark(function u(){var O,x;return m()().wrap(function(R){for(;;)switch(R.prev=R.next){case 0:return l(!0),R.next=3,c.Z.getSystemVersion();case 3:O=R.sent,x=O.data,x.retcode===0&&(a(x.data),l(!1));case 6:case"end":return R.stop()}},u)})),[]);return{fetchSystemVersion:_,version:o,loading:r}},L=function(){var t=(0,n.useState)({}),i=f()(t,2),o=i[0],a=i[1],A=(0,n.useState)(!1),p=f()(A,2),r=p[0],l=p[1],_=(0,n.useCallback)(v()(m()().mark(function u(){var O,x;return m()().wrap(function(R){for(;;)switch(R.prev=R.next){case 0:return l(!0),R.next=3,c.Z.getSystemStatus();case 3:O=R.sent,x=O.data,x.retcode===0&&(a(x.data),l(!1));case 6:case"end":return R.stop()}},u)})),[]);return{systemStatus:o,fetchSystemStatus:_,loading:r}}},33041:function(h,d,e){"use strict";e.r(d),e.d(d,{default:function(){return M}});var s=e(86250),f=e(6589),E=e(40169),m=e(65534),g=e(68508),v=e(62435),c=e(90194),D=e(2039),n=e(79495),P={sideBarWrapper:"sideBarWrapper___pApYb",version:"version___uhL2R"},y=e(86074),S=function(){var I=(0,f.useNavigate)(),L=(0,m.Uu)(),C=(0,n.aU)(),t=(0,D.qM)("setting"),i=t.t,o=(0,n.nv)(),a=o.version,A=o.fetchSystemVersion;(0,v.useEffect)(function(){location.host!==E.qp&&A()},[A]);function p(u,O,x,W,R){return{key:O,icon:x,children:W,label:(0,y.jsxs)(s.Z,{justify:"space-between",children:[i(u),(0,y.jsx)("span",{className:P.version,children:u==="system"&&a})]}),type:R}}var r=Object.values(c.qh).map(function(u){return p(u,u,c.Dr[u])}),l=function(O){var x=O.key;x===c.qh.Logout?C():I("/".concat(c.H7,"/").concat(x))},_=(0,v.useMemo)(function(){return[L]},[L]);return(0,y.jsx)("section",{className:P.sideBarWrapper,children:(0,y.jsx)(g.Z,{selectedKeys:_,mode:"inline",items:r,onClick:l,style:{width:312}})})},b=S,T=e(75041),U=function(){return(0,y.jsxs)(s.Z,{className:T.Z.settingWrapper,children:[(0,y.jsx)(b,{}),(0,y.jsx)(s.Z,{flex:1,className:T.Z.outletWrapper,children:(0,y.jsx)(f.Outlet,{})})]})},M=U},75041:function(h,d){"use strict";d.Z={settingWrapper:"settingWrapper___zesBR",outletWrapper:"outletWrapper___Zl2jj",itemDescription:"itemDescription___WJ1sc"}},86250:function(h,d,e){"use strict";e.d(d,{Z:function(){return p}});var s=e(62435),f=e(93967),E=e.n(f),m=e(98423),g=e(98065),v=e(53124),c=e(91945),D=e(45503);const n=["wrap","nowrap","wrap-reverse"],P=["flex-start","flex-end","start","end","center","space-between","space-around","space-evenly","stretch","normal","left","right"],y=["center","start","end","flex-start","flex-end","self-start","self-end","baseline","normal","stretch"],S=(r,l)=>{const _={};return n.forEach(u=>{_[`${r}-wrap-${u}`]=l.wrap===u}),_},b=(r,l)=>{const _={};return y.forEach(u=>{_[`${r}-align-${u}`]=l.align===u}),_[`${r}-align-stretch`]=!l.align&&!!l.vertical,_},T=(r,l)=>{const _={};return P.forEach(u=>{_[`${r}-justify-${u}`]=l.justify===u}),_};function U(r,l){return E()(Object.assign(Object.assign(Object.assign({},S(r,l)),b(r,l)),T(r,l)))}var M=U;const j=r=>{const{componentCls:l}=r;return{[l]:{display:"flex","&-vertical":{flexDirection:"column"},"&-rtl":{direction:"rtl"},"&:empty":{display:"none"}}}},I=r=>{const{componentCls:l}=r;return{[l]:{"&-gap-small":{gap:r.flexGapSM},"&-gap-middle":{gap:r.flexGap},"&-gap-large":{gap:r.flexGapLG}}}},L=r=>{const{componentCls:l}=r,_={};return n.forEach(u=>{_[`${l}-wrap-${u}`]={flexWrap:u}}),_},C=r=>{const{componentCls:l}=r,_={};return y.forEach(u=>{_[`${l}-align-${u}`]={alignItems:u}}),_},t=r=>{const{componentCls:l}=r,_={};return P.forEach(u=>{_[`${l}-justify-${u}`]={justifyContent:u}}),_},i=()=>({});var o=(0,c.I$)("Flex",r=>{const{paddingXS:l,padding:_,paddingLG:u}=r,O=(0,D.TS)(r,{flexGapSM:l,flexGap:_,flexGapLG:u});return[j(O),I(O),L(O),C(O),t(O)]},i,{resetStyle:!1}),a=function(r,l){var _={};for(var u in r)Object.prototype.hasOwnProperty.call(r,u)&&l.indexOf(u)<0&&(_[u]=r[u]);if(r!=null&&typeof Object.getOwnPropertySymbols=="function")for(var O=0,u=Object.getOwnPropertySymbols(r);O<u.length;O++)l.indexOf(u[O])<0&&Object.prototype.propertyIsEnumerable.call(r,u[O])&&(_[u[O]]=r[u[O]]);return _},p=s.forwardRef((r,l)=>{const{prefixCls:_,rootClassName:u,className:O,style:x,flex:W,gap:R,children:w,vertical:K=!1,component:B="div"}=r,F=a(r,["prefixCls","rootClassName","className","style","flex","gap","children","vertical","component"]),{flex:$,direction:Z,getPrefixCls:z}=s.useContext(v.E_),N=z("flex",_),[H,V,J]=o(N),Q=K!=null?K:$==null?void 0:$.vertical,X=E()(O,u,$==null?void 0:$.className,N,V,J,M(N,r),{[`${N}-rtl`]:Z==="rtl",[`${N}-gap-${R}`]:(0,g.n)(R),[`${N}-vertical`]:Q}),G=Object.assign(Object.assign({},$==null?void 0:$.style),x);return W&&(G.flex=W),R&&!(0,g.n)(R)&&(G.gap=R),H(s.createElement(B,Object.assign({ref:l,className:X,style:G},(0,m.Z)(F,["justify","wrap","align"])),w))})},33507:function(h,d){"use strict";const e=s=>({[s.componentCls]:{[`${s.antCls}-motion-collapse-legacy`]:{overflow:"hidden","&-active":{transition:`height ${s.motionDurationMid} ${s.motionEaseInOut},
        opacity ${s.motionDurationMid} ${s.motionEaseInOut} !important`}},[`${s.antCls}-motion-collapse`]:{overflow:"hidden",transition:`height ${s.motionDurationMid} ${s.motionEaseInOut},
        opacity ${s.motionDurationMid} ${s.motionEaseInOut} !important`}}});d.Z=e},88668:function(h,d,e){var s=e(83369),f=e(90619),E=e(72385);function m(g){var v=-1,c=g==null?0:g.length;for(this.__data__=new s;++v<c;)this.add(g[v])}m.prototype.add=m.prototype.push=f,m.prototype.has=E,h.exports=m},82908:function(h){function d(e,s){for(var f=-1,E=e==null?0:e.length;++f<E;)if(s(e[f],f,e))return!0;return!1}h.exports=d},90939:function(h,d,e){var s=e(2492),f=e(37005);function E(m,g,v,c,D){return m===g?!0:m==null||g==null||!f(m)&&!f(g)?m!==m&&g!==g:s(m,g,v,c,E,D)}h.exports=E},2492:function(h,d,e){var s=e(46384),f=e(67114),E=e(18351),m=e(16096),g=e(64160),v=e(1469),c=e(44144),D=e(36719),n=1,P="[object Arguments]",y="[object Array]",S="[object Object]",b=Object.prototype,T=b.hasOwnProperty;function U(M,j,I,L,C,t){var i=v(M),o=v(j),a=i?y:g(M),A=o?y:g(j);a=a==P?S:a,A=A==P?S:A;var p=a==S,r=A==S,l=a==A;if(l&&c(M)){if(!c(j))return!1;i=!0,p=!1}if(l&&!p)return t||(t=new s),i||D(M)?f(M,j,I,L,C,t):E(M,j,a,I,L,C,t);if(!(I&n)){var _=p&&T.call(M,"__wrapped__"),u=r&&T.call(j,"__wrapped__");if(_||u){var O=_?M.value():M,x=u?j.value():j;return t||(t=new s),C(O,x,I,L,t)}}return l?(t||(t=new s),m(M,j,I,L,C,t)):!1}h.exports=U},74757:function(h){function d(e,s){return e.has(s)}h.exports=d},67114:function(h,d,e){var s=e(88668),f=e(82908),E=e(74757),m=1,g=2;function v(c,D,n,P,y,S){var b=n&m,T=c.length,U=D.length;if(T!=U&&!(b&&U>T))return!1;var M=S.get(c),j=S.get(D);if(M&&j)return M==D&&j==c;var I=-1,L=!0,C=n&g?new s:void 0;for(S.set(c,D),S.set(D,c);++I<T;){var t=c[I],i=D[I];if(P)var o=b?P(i,t,I,D,c,S):P(t,i,I,c,D,S);if(o!==void 0){if(o)continue;L=!1;break}if(C){if(!f(D,function(a,A){if(!E(C,A)&&(t===a||y(t,a,n,P,S)))return C.push(A)})){L=!1;break}}else if(!(t===i||y(t,i,n,P,S))){L=!1;break}}return S.delete(c),S.delete(D),L}h.exports=v},18351:function(h,d,e){var s=e(62705),f=e(11149),E=e(77813),m=e(67114),g=e(68776),v=e(21814),c=1,D=2,n="[object Boolean]",P="[object Date]",y="[object Error]",S="[object Map]",b="[object Number]",T="[object RegExp]",U="[object Set]",M="[object String]",j="[object Symbol]",I="[object ArrayBuffer]",L="[object DataView]",C=s?s.prototype:void 0,t=C?C.valueOf:void 0;function i(o,a,A,p,r,l,_){switch(A){case L:if(o.byteLength!=a.byteLength||o.byteOffset!=a.byteOffset)return!1;o=o.buffer,a=a.buffer;case I:return!(o.byteLength!=a.byteLength||!l(new f(o),new f(a)));case n:case P:case b:return E(+o,+a);case y:return o.name==a.name&&o.message==a.message;case T:case M:return o==a+"";case S:var u=g;case U:var O=p&c;if(u||(u=v),o.size!=a.size&&!O)return!1;var x=_.get(o);if(x)return x==a;p|=D,_.set(o,a);var W=m(u(o),u(a),p,r,l,_);return _.delete(o),W;case j:if(t)return t.call(o)==t.call(a)}return!1}h.exports=i},16096:function(h,d,e){var s=e(58234),f=1,E=Object.prototype,m=E.hasOwnProperty;function g(v,c,D,n,P,y){var S=D&f,b=s(v),T=b.length,U=s(c),M=U.length;if(T!=M&&!S)return!1;for(var j=T;j--;){var I=b[j];if(!(S?I in c:m.call(c,I)))return!1}var L=y.get(v),C=y.get(c);if(L&&C)return L==c&&C==v;var t=!0;y.set(v,c),y.set(c,v);for(var i=S;++j<T;){I=b[j];var o=v[I],a=c[I];if(n)var A=S?n(a,o,I,c,v,y):n(o,a,I,v,c,y);if(!(A===void 0?o===a||P(o,a,D,n,y):A)){t=!1;break}i||(i=I=="constructor")}if(t&&!i){var p=v.constructor,r=c.constructor;p!=r&&"constructor"in v&&"constructor"in c&&!(typeof p=="function"&&p instanceof p&&typeof r=="function"&&r instanceof r)&&(t=!1)}return y.delete(v),y.delete(c),t}h.exports=g},68776:function(h){function d(e){var s=-1,f=Array(e.size);return e.forEach(function(E,m){f[++s]=[m,E]}),f}h.exports=d},90619:function(h){var d="__lodash_hash_undefined__";function e(s){return this.__data__.set(s,d),this}h.exports=e},72385:function(h){function d(e){return this.__data__.has(e)}h.exports=d},21814:function(h){function d(e){var s=-1,f=Array(e.size);return e.forEach(function(E){f[++s]=E}),f}h.exports=d},18446:function(h,d,e){var s=e(90939);function f(E,m){return s(E,m)}h.exports=f},86635:function(h){function d(e){return Promise.resolve().then(function(){var s=new Error("Cannot find module '"+e+"'");throw s.code="MODULE_NOT_FOUND",s})}d.keys=function(){return[]},d.resolve=d,d.id=86635,h.exports=d}}]);

//# sourceMappingURL=p__user-setting__index.c4d71bab.async.js.map