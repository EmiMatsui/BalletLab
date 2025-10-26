import cv2, numpy as np, mediapipe as mp
POSE_CONNS=list(mp.solutions.pose.POSE_CONNECTIONS)

def _draw_skeleton(img,kps,color=(0,255,0)):
    h,w=img.shape[:2]
    pts=[(int(x*w),int(y*h),v) for x,y,_,v in kps]
    for a,b in POSE_CONNS:
        xa,ya,va=pts[a]; xb,yb,vb=pts[b]
        if va>0.5 and vb>0.5: cv2.line(img,(xa,ya),(xb,yb),color,2)
    for x,y,v in pts:
        if v>0.5: cv2.circle(img,(x,y),3,color,-1)

def render_side_by_side(ideal_video_path,user_video_path,ideal_kps,user_kps,mapping,per_frame_results,out_path,stride=2,target_height=480,show=False):
    capI,capU=cv2.VideoCapture(ideal_video_path),cv2.VideoCapture(user_video_path)
    if not capI.isOpened() or not capU.isOpened(): return False
    fps=capU.get(cv2.CAP_PROP_FPS) or 30
    ret,fu=capU.read(); ret2,fi=capI.read()
    if not ret or not ret2: return False
    def resize_h(img,h=target_height):
        r=h/img.shape[0]; return cv2.resize(img,(int(img.shape[1]*r),h))
    fu,fi=resize_h(fu),resize_h(fi)
    pane_w=max(fu.shape[1],fi.shape[1])
    out=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps/(stride), (pane_w*2,target_height))
    for j in range(len(user_kps)):
        u_idx=j*stride; i_idx=int(mapping[j])*stride
        capU.set(cv2.CAP_PROP_POS_FRAMES,u_idx)
        capI.set(cv2.CAP_PROP_POS_FRAMES,i_idx)
        rU,fu=capU.read(); rI,fi=capI.read()
        if not rU or not rI: break
        fu,fi=resize_h(fu),resize_h(fi)
        _draw_skeleton(fi,ideal_kps[int(mapping[j])],(255,200,0))
        _draw_skeleton(fu,user_kps[j],(50,255,100))
        sc=per_frame_results[j]['score']
        txt=f"{sc:5.1f}"
        cv2.putText(fu,f"Score: {txt}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        combo=np.concatenate([fi,fu],axis=1)
        if show:
            cv2.imshow("compare",combo)
            if cv2.waitKey(1)&0xFF==27: break
        out.write(combo)
    out.release(); capI.release(); capU.release()
    if show: cv2.destroyAllWindows()
    print(f"✅ 書き出し: {out_path}")
    return True
