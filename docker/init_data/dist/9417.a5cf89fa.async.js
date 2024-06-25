"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[9417],{89705:function(vn,rt,v){v.d(rt,{Z:function(){return at}});var $=v(87462),I=v(62435),O={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M176 511a56 56 0 10112 0 56 56 0 10-112 0zm280 0a56 56 0 10112 0 56 56 0 10-112 0zm280 0a56 56 0 10112 0 56 56 0 10-112 0z"}}]},name:"ellipsis",theme:"outlined"},ae=O,w=v(84089),F=function(ve,Ie){return I.createElement(w.Z,(0,$.Z)({},ve,{ref:Ie,icon:ae}))},at=I.forwardRef(F)},72512:function(vn,rt,v){v.d(rt,{iz:function(){return pt},ck:function(){return De},BW:function(){return mt},sN:function(){return De},Wd:function(){return Ye},ZP:function(){return ur},Xl:function(){return Me}});var $=v(87462),I=v(4942),O=v(1413),ae=v(74902),w=v(97685),F=v(45987),at=v(93967),oe=v.n(at),ve=v(39983),Ie=v(21770),xt=v(91881),Et=v(80334),r=v(62435),dn=v(61254),Kt=r.createContext(null);function Nt(t,e){return t===void 0?null:"".concat(t,"-").concat(e)}function Ot(t){var e=r.useContext(Kt);return Nt(e,t)}var fn=v(56982),mn=["children","locked"],J=r.createContext(null);function pn(t,e){var a=(0,O.Z)({},t);return Object.keys(e).forEach(function(i){var n=e[i];n!==void 0&&(a[i]=n)}),a}function Ke(t){var e=t.children,a=t.locked,i=(0,F.Z)(t,mn),n=r.useContext(J),o=(0,fn.Z)(function(){return pn(n,i)},[n,i],function(l,u){return!a&&(l[0]!==u[0]||!(0,xt.Z)(l[1],u[1],!0))});return r.createElement(J.Provider,{value:o},e)}var hn=[],wt=r.createContext(null);function Ge(){return r.useContext(wt)}var At=r.createContext(hn);function Me(t){var e=r.useContext(At);return r.useMemo(function(){return t!==void 0?[].concat((0,ae.Z)(e),[t]):e},[e,t])}var Dt=r.createContext(null),gn=r.createContext({}),it=gn,Cn=v(5110);function Lt(t){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!1;if((0,Cn.Z)(t)){var a=t.nodeName.toLowerCase(),i=["input","select","textarea","button"].includes(a)||t.isContentEditable||a==="a"&&!!t.getAttribute("href"),n=t.getAttribute("tabindex"),o=Number(n),l=null;return n&&!Number.isNaN(o)?l=o:i&&l===null&&(l=0),i&&t.disabled&&(l=null),l!==null&&(l>=0||e&&l<0)}return!1}function $t(t){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!1,a=(0,ae.Z)(t.querySelectorAll("*")).filter(function(i){return Lt(i,e)});return Lt(t,e)&&a.unshift(t),a}var We=null;function Tr(){We=document.activeElement}function Fr(){We=null}function Vr(){if(We)try{We.focus()}catch(t){}}function zr(t,e){if(e.keyCode===9){var a=$t(t),i=a[e.shiftKey?0:a.length-1],n=i===document.activeElement||t===document.activeElement;if(n){var o=a[e.shiftKey?a.length-1:0];o.focus(),e.preventDefault()}}}var ie=v(15105),Ne=v(75164),lt=ie.Z.LEFT,ot=ie.Z.RIGHT,ut=ie.Z.UP,Be=ie.Z.DOWN,He=ie.Z.ENTER,kt=ie.Z.ESC,Oe=ie.Z.HOME,we=ie.Z.END,Tt=[ut,Be,lt,ot];function yn(t,e,a,i){var n,o,l,u,c="prev",s="next",C="children",d="parent";if(t==="inline"&&i===He)return{inlineTrigger:!0};var b=(n={},(0,I.Z)(n,ut,c),(0,I.Z)(n,Be,s),n),M=(o={},(0,I.Z)(o,lt,a?s:c),(0,I.Z)(o,ot,a?c:s),(0,I.Z)(o,Be,C),(0,I.Z)(o,He,C),o),p=(l={},(0,I.Z)(l,ut,c),(0,I.Z)(l,Be,s),(0,I.Z)(l,He,C),(0,I.Z)(l,kt,d),(0,I.Z)(l,lt,a?C:d),(0,I.Z)(l,ot,a?d:C),l),K={inline:b,horizontal:M,vertical:p,inlineSub:b,horizontalSub:p,verticalSub:p},x=(u=K["".concat(t).concat(e?"":"Sub")])===null||u===void 0?void 0:u[i];switch(x){case c:return{offset:-1,sibling:!0};case s:return{offset:1,sibling:!0};case d:return{offset:-1,sibling:!1};case C:return{offset:1,sibling:!1};default:return null}}function bn(t){for(var e=t;e;){if(e.getAttribute("data-menu-list"))return e;e=e.parentElement}return null}function In(t,e){for(var a=t||document.activeElement;a;){if(e.has(a))return a;a=a.parentElement}return null}function st(t,e){var a=$t(t,!0);return a.filter(function(i){return e.has(i)})}function Ft(t,e,a){var i=arguments.length>3&&arguments[3]!==void 0?arguments[3]:1;if(!t)return null;var n=st(t,e),o=n.length,l=n.findIndex(function(u){return a===u});return i<0?l===-1?l=o-1:l-=1:i>0&&(l+=1),l=(l+o)%o,n[l]}var ct=function(e,a){var i=new Set,n=new Map,o=new Map;return e.forEach(function(l){var u=document.querySelector("[data-menu-id='".concat(Nt(a,l),"']"));u&&(i.add(u),o.set(u,l),n.set(l,u))}),{elements:i,key2element:n,element2key:o}};function Mn(t,e,a,i,n,o,l,u,c,s){var C=r.useRef(),d=r.useRef();d.current=e;var b=function(){Ne.Z.cancel(C.current)};return r.useEffect(function(){return function(){b()}},[]),function(M){var p=M.which;if([].concat(Tt,[He,kt,Oe,we]).includes(p)){var K=o(),x=ct(K,i),N=x,h=N.elements,m=N.key2element,f=N.element2key,Z=m.get(e),y=In(Z,h),A=f.get(y),S=yn(t,l(A,!0).length===1,a,p);if(!S&&p!==Oe&&p!==we)return;(Tt.includes(p)||[Oe,we].includes(p))&&M.preventDefault();var Q=function(V){if(V){var q=V,_=V.querySelector("a");_!=null&&_.getAttribute("href")&&(q=_);var ee=f.get(V);u(ee),b(),C.current=(0,Ne.Z)(function(){d.current===ee&&q.focus()})}};if([Oe,we].includes(p)||S.sibling||!y){var P;!y||t==="inline"?P=n.current:P=bn(y);var U,L=st(P,h);p===Oe?U=L[0]:p===we?U=L[L.length-1]:U=Ft(P,h,y,S.offset),Q(U)}else if(S.inlineTrigger)c(A);else if(S.offset>0)c(A,!0),b(),C.current=(0,Ne.Z)(function(){x=ct(K,i);var le=y.getAttribute("aria-controls"),V=document.getElementById(le),q=Ft(V,x.elements);Q(q)},5);else if(S.offset<0){var k=l(A,!0),B=k[k.length-2],H=m.get(B);c(B,!1),Q(H)}}s==null||s(M)}}function Sn(t){Promise.resolve().then(t)}var vt="__RC_UTIL_PATH_SPLIT__",Vt=function(e){return e.join(vt)},Zn=function(e){return e.split(vt)},dt="rc-menu-more";function Rn(){var t=r.useState({}),e=(0,w.Z)(t,2),a=e[1],i=(0,r.useRef)(new Map),n=(0,r.useRef)(new Map),o=r.useState([]),l=(0,w.Z)(o,2),u=l[0],c=l[1],s=(0,r.useRef)(0),C=(0,r.useRef)(!1),d=function(){C.current||a({})},b=(0,r.useCallback)(function(m,f){var Z=Vt(f);n.current.set(Z,m),i.current.set(m,Z),s.current+=1;var y=s.current;Sn(function(){y===s.current&&d()})},[]),M=(0,r.useCallback)(function(m,f){var Z=Vt(f);n.current.delete(Z),i.current.delete(m)},[]),p=(0,r.useCallback)(function(m){c(m)},[]),K=(0,r.useCallback)(function(m,f){var Z=i.current.get(m)||"",y=Zn(Z);return f&&u.includes(y[0])&&y.unshift(dt),y},[u]),x=(0,r.useCallback)(function(m,f){return m.some(function(Z){var y=K(Z,!0);return y.includes(f)})},[K]),N=function(){var f=(0,ae.Z)(i.current.keys());return u.length&&f.push(dt),f},h=(0,r.useCallback)(function(m){var f="".concat(i.current.get(m)).concat(vt),Z=new Set;return(0,ae.Z)(n.current.keys()).forEach(function(y){y.startsWith(f)&&Z.add(n.current.get(y))}),Z},[]);return r.useEffect(function(){return function(){C.current=!0}},[]),{registerPath:b,unregisterPath:M,refreshOverflowKeys:p,isSubPathKey:x,getKeyPath:K,getKeys:N,getSubPathKeys:h}}function Ae(t){var e=r.useRef(t);e.current=t;var a=r.useCallback(function(){for(var i,n=arguments.length,o=new Array(n),l=0;l<n;l++)o[l]=arguments[l];return(i=e.current)===null||i===void 0?void 0:i.call.apply(i,[e].concat(o))},[]);return t?a:void 0}var Pn=Math.random().toFixed(5).toString().slice(2),zt=0;function xn(t){var e=(0,Ie.Z)(t,{value:t}),a=(0,w.Z)(e,2),i=a[0],n=a[1];return r.useEffect(function(){zt+=1;var o="".concat(Pn,"-").concat(zt);n("rc-menu-uuid-".concat(o))},[]),i}var En=v(15671),Kn=v(43144),Nn=v(60136),On=v(73568),Ut=v(98423),wn=v(42550);function Gt(t,e,a,i){var n=r.useContext(J),o=n.activeKey,l=n.onActive,u=n.onInactive,c={active:o===t};return e||(c.onMouseEnter=function(s){a==null||a({key:t,domEvent:s}),l(t)},c.onMouseLeave=function(s){i==null||i({key:t,domEvent:s}),u(t)}),c}function Wt(t){var e=r.useContext(J),a=e.mode,i=e.rtl,n=e.inlineIndent;if(a!=="inline")return null;var o=t;return i?{paddingRight:o*n}:{paddingLeft:o*n}}function Bt(t){var e=t.icon,a=t.props,i=t.children,n;return e===null||e===!1?null:(typeof e=="function"?n=r.createElement(e,(0,O.Z)({},a)):typeof e!="boolean"&&(n=e),n||i||null)}var An=["item"];function je(t){var e=t.item,a=(0,F.Z)(t,An);return Object.defineProperty(a,"item",{get:function(){return(0,Et.ZP)(!1,"`info.item` is deprecated since we will move to function component that not provides React Node instance in future."),e}}),a}var Dn=["title","attribute","elementRef"],Ln=["style","className","eventKey","warnKey","disabled","itemIcon","children","role","onMouseEnter","onMouseLeave","onClick","onKeyDown","onFocus"],$n=["active"],kn=function(t){(0,Nn.Z)(a,t);var e=(0,On.Z)(a);function a(){return(0,En.Z)(this,a),e.apply(this,arguments)}return(0,Kn.Z)(a,[{key:"render",value:function(){var n=this.props,o=n.title,l=n.attribute,u=n.elementRef,c=(0,F.Z)(n,Dn),s=(0,Ut.Z)(c,["eventKey","popupClassName","popupOffset","onTitleClick"]);return(0,Et.ZP)(!l,"`attribute` of Menu.Item is deprecated. Please pass attribute directly."),r.createElement(ve.Z.Item,(0,$.Z)({},l,{title:typeof o=="string"?o:void 0},s,{ref:u}))}}]),a}(r.Component),Tn=r.forwardRef(function(t,e){var a,i=t.style,n=t.className,o=t.eventKey,l=t.warnKey,u=t.disabled,c=t.itemIcon,s=t.children,C=t.role,d=t.onMouseEnter,b=t.onMouseLeave,M=t.onClick,p=t.onKeyDown,K=t.onFocus,x=(0,F.Z)(t,Ln),N=Ot(o),h=r.useContext(J),m=h.prefixCls,f=h.onItemClick,Z=h.disabled,y=h.overflowDisabled,A=h.itemIcon,S=h.selectedKeys,Q=h.onActive,P=r.useContext(it),U=P._internalRenderMenuItem,L="".concat(m,"-item"),k=r.useRef(),B=r.useRef(),H=Z||u,le=(0,wn.x1)(e,B),V=Me(o),q=function(T){return{key:o,keyPath:(0,ae.Z)(V).reverse(),item:k.current,domEvent:T}},_=c||A,ee=Gt(o,H,d,b),de=ee.active,fe=(0,F.Z)(ee,$n),ue=S.includes(o),me=Wt(V.length),pe=function(T){if(!H){var te=q(T);M==null||M(je(te)),f(te)}},z=function(T){if(p==null||p(T),T.which===ie.Z.ENTER){var te=q(T);M==null||M(je(te)),f(te)}},j=function(T){Q(o),K==null||K(T)},Ze={};t.role==="option"&&(Ze["aria-selected"]=ue);var he=r.createElement(kn,(0,$.Z)({ref:k,elementRef:le,role:C===null?"none":C||"menuitem",tabIndex:u?null:-1,"data-menu-id":y&&N?null:N},x,fe,Ze,{component:"li","aria-disabled":u,style:(0,O.Z)((0,O.Z)({},me),i),className:oe()(L,(a={},(0,I.Z)(a,"".concat(L,"-active"),de),(0,I.Z)(a,"".concat(L,"-selected"),ue),(0,I.Z)(a,"".concat(L,"-disabled"),H),a),n),onClick:pe,onKeyDown:z,onFocus:j}),s,r.createElement(Bt,{props:(0,O.Z)((0,O.Z)({},t),{},{isSelected:ue}),icon:_}));return U&&(he=U(he,t,{selected:ue})),he});function Fn(t,e){var a=t.eventKey,i=Ge(),n=Me(a);return r.useEffect(function(){if(i)return i.registerPath(a,n),function(){i.unregisterPath(a,n)}},[n]),i?null:r.createElement(Tn,(0,$.Z)({},t,{ref:e}))}var De=r.forwardRef(Fn),Vn=["className","children"],zn=function(e,a){var i=e.className,n=e.children,o=(0,F.Z)(e,Vn),l=r.useContext(J),u=l.prefixCls,c=l.mode,s=l.rtl;return r.createElement("ul",(0,$.Z)({className:oe()(u,s&&"".concat(u,"-rtl"),"".concat(u,"-sub"),"".concat(u,"-").concat(c==="inline"?"inline":"vertical"),i),role:"menu"},o,{"data-menu-list":!0,ref:a}),n)},Ht=r.forwardRef(zn);Ht.displayName="SubMenuList";var jt=Ht,Un=v(50344);function ft(t,e){return(0,Un.Z)(t).map(function(a,i){if(r.isValidElement(a)){var n,o,l=a.key,u=(n=(o=a.props)===null||o===void 0?void 0:o.eventKey)!==null&&n!==void 0?n:l,c=u==null;c&&(u="tmp_key-".concat([].concat((0,ae.Z)(e),[i]).join("-")));var s={key:u,eventKey:u};return r.cloneElement(a,s)}return a})}var Gn=v(40228),D={adjustX:1,adjustY:1},Wn={topLeft:{points:["bl","tl"],overflow:D},topRight:{points:["br","tr"],overflow:D},bottomLeft:{points:["tl","bl"],overflow:D},bottomRight:{points:["tr","br"],overflow:D},leftTop:{points:["tr","tl"],overflow:D},leftBottom:{points:["br","bl"],overflow:D},rightTop:{points:["tl","tr"],overflow:D},rightBottom:{points:["bl","br"],overflow:D}},Bn={topLeft:{points:["bl","tl"],overflow:D},topRight:{points:["br","tr"],overflow:D},bottomLeft:{points:["tl","bl"],overflow:D},bottomRight:{points:["tr","br"],overflow:D},rightTop:{points:["tr","tl"],overflow:D},rightBottom:{points:["br","bl"],overflow:D},leftTop:{points:["tl","tr"],overflow:D},leftBottom:{points:["bl","br"],overflow:D}},Ur=null;function Yt(t,e,a){if(e)return e;if(a)return a[t]||a.other}var Hn={horizontal:"bottomLeft",vertical:"rightTop","vertical-left":"rightTop","vertical-right":"leftTop"};function jn(t){var e=t.prefixCls,a=t.visible,i=t.children,n=t.popup,o=t.popupStyle,l=t.popupClassName,u=t.popupOffset,c=t.disabled,s=t.mode,C=t.onVisibleChange,d=r.useContext(J),b=d.getPopupContainer,M=d.rtl,p=d.subMenuOpenDelay,K=d.subMenuCloseDelay,x=d.builtinPlacements,N=d.triggerSubMenuAction,h=d.forceSubMenuRender,m=d.rootClassName,f=d.motion,Z=d.defaultMotions,y=r.useState(!1),A=(0,w.Z)(y,2),S=A[0],Q=A[1],P=M?(0,O.Z)((0,O.Z)({},Bn),x):(0,O.Z)((0,O.Z)({},Wn),x),U=Hn[s],L=Yt(s,f,Z),k=r.useRef(L);s!=="inline"&&(k.current=L);var B=(0,O.Z)((0,O.Z)({},k.current),{},{leavedClassName:"".concat(e,"-hidden"),removeOnLeave:!1,motionAppear:!0}),H=r.useRef();return r.useEffect(function(){return H.current=(0,Ne.Z)(function(){Q(a)}),function(){Ne.Z.cancel(H.current)}},[a]),r.createElement(Gn.Z,{prefixCls:e,popupClassName:oe()("".concat(e,"-popup"),(0,I.Z)({},"".concat(e,"-rtl"),M),l,m),stretch:s==="horizontal"?"minWidth":null,getPopupContainer:b,builtinPlacements:P,popupPlacement:U,popupVisible:S,popup:n,popupStyle:o,popupAlign:u&&{offset:u},action:c?[]:[N],mouseEnterDelay:p,mouseLeaveDelay:K,onPopupVisibleChange:C,forceRender:h,popupMotion:B,fresh:!0},i)}var Yn=v(82225);function Xn(t){var e=t.id,a=t.open,i=t.keyPath,n=t.children,o="inline",l=r.useContext(J),u=l.prefixCls,c=l.forceSubMenuRender,s=l.motion,C=l.defaultMotions,d=l.mode,b=r.useRef(!1);b.current=d===o;var M=r.useState(!b.current),p=(0,w.Z)(M,2),K=p[0],x=p[1],N=b.current?a:!1;r.useEffect(function(){b.current&&x(!1)},[d]);var h=(0,O.Z)({},Yt(o,s,C));i.length>1&&(h.motionAppear=!1);var m=h.onVisibleChanged;return h.onVisibleChanged=function(f){return!b.current&&!f&&x(!0),m==null?void 0:m(f)},K?null:r.createElement(Ke,{mode:o,locked:!b.current},r.createElement(Yn.ZP,(0,$.Z)({visible:N},h,{forceRender:c,removeOnLeave:!1,leavedClassName:"".concat(u,"-hidden")}),function(f){var Z=f.className,y=f.style;return r.createElement(jt,{id:e,className:Z,style:y},n)}))}var Jn=["style","className","title","eventKey","warnKey","disabled","internalPopupClose","children","itemIcon","expandIcon","popupClassName","popupOffset","popupStyle","onClick","onMouseEnter","onMouseLeave","onTitleClick","onTitleMouseEnter","onTitleMouseLeave"],Qn=["active"],qn=function(e){var a,i=e.style,n=e.className,o=e.title,l=e.eventKey,u=e.warnKey,c=e.disabled,s=e.internalPopupClose,C=e.children,d=e.itemIcon,b=e.expandIcon,M=e.popupClassName,p=e.popupOffset,K=e.popupStyle,x=e.onClick,N=e.onMouseEnter,h=e.onMouseLeave,m=e.onTitleClick,f=e.onTitleMouseEnter,Z=e.onTitleMouseLeave,y=(0,F.Z)(e,Jn),A=Ot(l),S=r.useContext(J),Q=S.prefixCls,P=S.mode,U=S.openKeys,L=S.disabled,k=S.overflowDisabled,B=S.activeKey,H=S.selectedKeys,le=S.itemIcon,V=S.expandIcon,q=S.onItemClick,_=S.onOpenChange,ee=S.onActive,de=r.useContext(it),fe=de._internalRenderSubMenuItem,ue=r.useContext(Dt),me=ue.isSubPathKey,pe=Me(),z="".concat(Q,"-submenu"),j=L||c,Ze=r.useRef(),he=r.useRef(),ge=d!=null?d:le,T=b!=null?b:V,te=U.includes(l),se=!k&&te,Xe=me(H,l),Re=Gt(l,j,f,Z),Ce=Re.active,gt=(0,F.Z)(Re,Qn),Xt=r.useState(!1),Ct=(0,w.Z)(Xt,2),$e=Ct[0],Je=Ct[1],Qe=function(X){j||Je(X)},ne=function(X){Qe(!0),N==null||N({key:l,domEvent:X})},yt=function(X){Qe(!1),h==null||h({key:l,domEvent:X})},ke=r.useMemo(function(){return Ce||(P!=="inline"?$e||me([B],l):!1)},[P,Ce,B,$e,l,me]),qe=Wt(pe.length),bt=function(X){j||(m==null||m({key:l,domEvent:X}),P==="inline"&&_(l,!te))},Pe=Ae(function(re){x==null||x(je(re)),q(re)}),Te=function(X){P!=="inline"&&_(l,X)},Fe=function(){ee(l)},Ve=A&&"".concat(A,"-popup"),xe=r.createElement("div",(0,$.Z)({role:"menuitem",style:qe,className:"".concat(z,"-title"),tabIndex:j?null:-1,ref:Ze,title:typeof o=="string"?o:null,"data-menu-id":k&&A?null:A,"aria-expanded":se,"aria-haspopup":!0,"aria-controls":Ve,"aria-disabled":j,onClick:bt,onFocus:Fe},gt),o,r.createElement(Bt,{icon:P!=="horizontal"?T:void 0,props:(0,O.Z)((0,O.Z)({},e),{},{isOpen:se,isSubMenu:!0})},r.createElement("i",{className:"".concat(z,"-arrow")}))),Y=r.useRef(P);if(P!=="inline"&&pe.length>1?Y.current="vertical":Y.current=P,!k){var Ee=Y.current;xe=r.createElement(jn,{mode:Ee,prefixCls:z,visible:!s&&se&&P!=="inline",popupClassName:M,popupOffset:p,popupStyle:K,popup:r.createElement(Ke,{mode:Ee==="horizontal"?"vertical":Ee},r.createElement(jt,{id:Ve,ref:he},C)),disabled:j,onVisibleChange:Te},xe)}var ye=r.createElement(ve.Z.Item,(0,$.Z)({role:"none"},y,{component:"li",style:i,className:oe()(z,"".concat(z,"-").concat(P),n,(a={},(0,I.Z)(a,"".concat(z,"-open"),se),(0,I.Z)(a,"".concat(z,"-active"),ke),(0,I.Z)(a,"".concat(z,"-selected"),Xe),(0,I.Z)(a,"".concat(z,"-disabled"),j),a)),onMouseEnter:ne,onMouseLeave:yt}),xe,!k&&r.createElement(Xn,{id:Ve,open:se,keyPath:pe},C));return fe&&(ye=fe(ye,e,{selected:Xe,active:ke,open:se,disabled:j})),r.createElement(Ke,{onItemClick:Pe,mode:P==="horizontal"?"vertical":P,itemIcon:ge,expandIcon:T},ye)};function Ye(t){var e=t.eventKey,a=t.children,i=Me(e),n=ft(a,i),o=Ge();r.useEffect(function(){if(o)return o.registerPath(e,i),function(){o.unregisterPath(e,i)}},[i]);var l;return o?l=n:l=r.createElement(qn,t,n),r.createElement(At.Provider,{value:i},l)}var _n=v(71002),er=["className","title","eventKey","children"],tr=["children"],nr=function(e){var a=e.className,i=e.title,n=e.eventKey,o=e.children,l=(0,F.Z)(e,er),u=r.useContext(J),c=u.prefixCls,s="".concat(c,"-item-group");return r.createElement("li",(0,$.Z)({role:"presentation"},l,{onClick:function(d){return d.stopPropagation()},className:oe()(s,a)}),r.createElement("div",{role:"presentation",className:"".concat(s,"-title"),title:typeof i=="string"?i:void 0},i),r.createElement("ul",{role:"group",className:"".concat(s,"-list")},o))};function mt(t){var e=t.children,a=(0,F.Z)(t,tr),i=Me(a.eventKey),n=ft(e,i),o=Ge();return o?n:r.createElement(nr,(0,Ut.Z)(a,["warnKey"]),n)}function pt(t){var e=t.className,a=t.style,i=r.useContext(J),n=i.prefixCls,o=Ge();return o?null:r.createElement("li",{role:"separator",className:oe()("".concat(n,"-item-divider"),e),style:a})}var rr=["label","children","key","type"];function ht(t){return(t||[]).map(function(e,a){if(e&&(0,_n.Z)(e)==="object"){var i=e,n=i.label,o=i.children,l=i.key,u=i.type,c=(0,F.Z)(i,rr),s=l!=null?l:"tmp-".concat(a);return o||u==="group"?u==="group"?r.createElement(mt,(0,$.Z)({key:s},c,{title:n}),ht(o)):r.createElement(Ye,(0,$.Z)({key:s},c,{title:n}),ht(o)):u==="divider"?r.createElement(pt,(0,$.Z)({key:s},c)):r.createElement(De,(0,$.Z)({key:s},c),n)}return null}).filter(function(e){return e})}function ar(t,e,a){var i=t;return e&&(i=ht(e)),ft(i,a)}var ir=["prefixCls","rootClassName","style","className","tabIndex","items","children","direction","id","mode","inlineCollapsed","disabled","disabledOverflow","subMenuOpenDelay","subMenuCloseDelay","forceSubMenuRender","defaultOpenKeys","openKeys","activeKey","defaultActiveFirst","selectable","multiple","defaultSelectedKeys","selectedKeys","onSelect","onDeselect","inlineIndent","motion","defaultMotions","triggerSubMenuAction","builtinPlacements","itemIcon","expandIcon","overflowedIndicator","overflowedIndicatorPopupClassName","getPopupContainer","onClick","onOpenChange","onKeyDown","openAnimation","openTransitionName","_internalRenderMenuItem","_internalRenderSubMenuItem"],Se=[],lr=r.forwardRef(function(t,e){var a,i,n=t,o=n.prefixCls,l=o===void 0?"rc-menu":o,u=n.rootClassName,c=n.style,s=n.className,C=n.tabIndex,d=C===void 0?0:C,b=n.items,M=n.children,p=n.direction,K=n.id,x=n.mode,N=x===void 0?"vertical":x,h=n.inlineCollapsed,m=n.disabled,f=n.disabledOverflow,Z=n.subMenuOpenDelay,y=Z===void 0?.1:Z,A=n.subMenuCloseDelay,S=A===void 0?.1:A,Q=n.forceSubMenuRender,P=n.defaultOpenKeys,U=n.openKeys,L=n.activeKey,k=n.defaultActiveFirst,B=n.selectable,H=B===void 0?!0:B,le=n.multiple,V=le===void 0?!1:le,q=n.defaultSelectedKeys,_=n.selectedKeys,ee=n.onSelect,de=n.onDeselect,fe=n.inlineIndent,ue=fe===void 0?24:fe,me=n.motion,pe=n.defaultMotions,z=n.triggerSubMenuAction,j=z===void 0?"hover":z,Ze=n.builtinPlacements,he=n.itemIcon,ge=n.expandIcon,T=n.overflowedIndicator,te=T===void 0?"...":T,se=n.overflowedIndicatorPopupClassName,Xe=n.getPopupContainer,Re=n.onClick,Ce=n.onOpenChange,gt=n.onKeyDown,Xt=n.openAnimation,Ct=n.openTransitionName,$e=n._internalRenderMenuItem,Je=n._internalRenderSubMenuItem,Qe=(0,F.Z)(n,ir),ne=r.useMemo(function(){return ar(M,b,Se)},[M,b]),yt=r.useState(!1),ke=(0,w.Z)(yt,2),qe=ke[0],bt=ke[1],Pe=r.useRef(),Te=xn(K),Fe=p==="rtl",Ve=(0,Ie.Z)(P,{value:U,postState:function(g){return g||Se}}),xe=(0,w.Z)(Ve,2),Y=xe[0],Ee=xe[1],ye=function(g){var R=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!1;function G(){Ee(g),Ce==null||Ce(g)}R?(0,dn.flushSync)(G):G()},re=r.useState(Y),X=(0,w.Z)(re,2),sr=X[0],cr=X[1],It=r.useRef(!1),vr=r.useMemo(function(){return(N==="inline"||N==="vertical")&&h?["vertical",h]:[N,!1]},[N,h]),Jt=(0,w.Z)(vr,2),_e=Jt[0],Mt=Jt[1],Qt=_e==="inline",dr=r.useState(_e),qt=(0,w.Z)(dr,2),ce=qt[0],fr=qt[1],mr=r.useState(Mt),_t=(0,w.Z)(mr,2),pr=_t[0],hr=_t[1];r.useEffect(function(){fr(_e),hr(Mt),It.current&&(Qt?Ee(sr):ye(Se))},[_e,Mt]);var gr=r.useState(0),en=(0,w.Z)(gr,2),et=en[0],Cr=en[1],St=et>=ne.length-1||ce!=="horizontal"||f;r.useEffect(function(){Qt&&cr(Y)},[Y]),r.useEffect(function(){return It.current=!0,function(){It.current=!1}},[]);var be=Rn(),tn=be.registerPath,nn=be.unregisterPath,yr=be.refreshOverflowKeys,rn=be.isSubPathKey,br=be.getKeyPath,an=be.getKeys,Ir=be.getSubPathKeys,Mr=r.useMemo(function(){return{registerPath:tn,unregisterPath:nn}},[tn,nn]),Sr=r.useMemo(function(){return{isSubPathKey:rn}},[rn]);r.useEffect(function(){yr(St?Se:ne.slice(et+1).map(function(E){return E.key}))},[et,St]);var Zr=(0,Ie.Z)(L||k&&((a=ne[0])===null||a===void 0?void 0:a.key),{value:L}),ln=(0,w.Z)(Zr,2),ze=ln[0],Zt=ln[1],Rr=Ae(function(E){Zt(E)}),Pr=Ae(function(){Zt(void 0)});(0,r.useImperativeHandle)(e,function(){return{list:Pe.current,focus:function(g){var R,G=an(),W=ct(G,Te),nt=W.elements,Rt=W.key2element,$r=W.element2key,sn=st(Pe.current,nt),cn=ze!=null?ze:sn[0]?$r.get(sn[0]):(R=ne.find(function(kr){return!kr.props.disabled}))===null||R===void 0?void 0:R.key,Ue=Rt.get(cn);if(cn&&Ue){var Pt;Ue==null||(Pt=Ue.focus)===null||Pt===void 0||Pt.call(Ue,g)}}}});var xr=(0,Ie.Z)(q||[],{value:_,postState:function(g){return Array.isArray(g)?g:g==null?Se:[g]}}),on=(0,w.Z)(xr,2),tt=on[0],Er=on[1],Kr=function(g){if(H){var R=g.key,G=tt.includes(R),W;V?G?W=tt.filter(function(Rt){return Rt!==R}):W=[].concat((0,ae.Z)(tt),[R]):W=[R],Er(W);var nt=(0,O.Z)((0,O.Z)({},g),{},{selectedKeys:W});G?de==null||de(nt):ee==null||ee(nt)}!V&&Y.length&&ce!=="inline"&&ye(Se)},Nr=Ae(function(E){Re==null||Re(je(E)),Kr(E)}),un=Ae(function(E,g){var R=Y.filter(function(W){return W!==E});if(g)R.push(E);else if(ce!=="inline"){var G=Ir(E);R=R.filter(function(W){return!G.has(W)})}(0,xt.Z)(Y,R,!0)||ye(R,!0)}),Or=function(g,R){var G=R!=null?R:!Y.includes(g);un(g,G)},wr=Mn(ce,ze,Fe,Te,Pe,an,br,Zt,Or,gt);r.useEffect(function(){bt(!0)},[]);var Ar=r.useMemo(function(){return{_internalRenderMenuItem:$e,_internalRenderSubMenuItem:Je}},[$e,Je]),Dr=ce!=="horizontal"||f?ne:ne.map(function(E,g){return r.createElement(Ke,{key:E.key,overflowDisabled:g>et},E)}),Lr=r.createElement(ve.Z,(0,$.Z)({id:K,ref:Pe,prefixCls:"".concat(l,"-overflow"),component:"ul",itemComponent:De,className:oe()(l,"".concat(l,"-root"),"".concat(l,"-").concat(ce),s,(i={},(0,I.Z)(i,"".concat(l,"-inline-collapsed"),pr),(0,I.Z)(i,"".concat(l,"-rtl"),Fe),i),u),dir:p,style:c,role:"menu",tabIndex:d,data:Dr,renderRawItem:function(g){return g},renderRawRest:function(g){var R=g.length,G=R?ne.slice(-R):null;return r.createElement(Ye,{eventKey:dt,title:te,disabled:St,internalPopupClose:R===0,popupClassName:se},G)},maxCount:ce!=="horizontal"||f?ve.Z.INVALIDATE:ve.Z.RESPONSIVE,ssr:"full","data-menu-list":!0,onVisibleChange:function(g){Cr(g)},onKeyDown:wr},Qe));return r.createElement(it.Provider,{value:Ar},r.createElement(Kt.Provider,{value:Te},r.createElement(Ke,{prefixCls:l,rootClassName:u,mode:ce,openKeys:Y,rtl:Fe,disabled:m,motion:qe?me:null,defaultMotions:qe?pe:null,activeKey:ze,onActive:Rr,onInactive:Pr,selectedKeys:tt,inlineIndent:ue,subMenuOpenDelay:y,subMenuCloseDelay:S,forceSubMenuRender:Q,builtinPlacements:Ze,triggerSubMenuAction:j,getPopupContainer:Xe,itemIcon:he,expandIcon:ge,onItemClick:Nr,onOpenChange:un},r.createElement(Dt.Provider,{value:Sr},Lr),r.createElement("div",{style:{display:"none"},"aria-hidden":!0},r.createElement(wt.Provider,{value:Mr},ne)))))}),or=lr,Le=or;Le.Item=De,Le.SubMenu=Ye,Le.ItemGroup=mt,Le.Divider=pt;var ur=Le}}]);

//# sourceMappingURL=9417.a5cf89fa.async.js.map