import numpy as np

def _angle(a,b,c):
    v1, v2 = a-b, c-b
    cos = np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8), -1,1)
    return np.degrees(np.arccos(cos))

def _joint_angles(p):
    J = {}
    L_SH,R_SH,L_EL,R_EL,L_WR,R_WR,L_HIP,R_HIP,L_KN,R_KN,L_AN,R_AN = 11,12,13,14,15,16,23,24,25,26,27,28
    J['L_elbow']=_angle(p[L_SH],p[L_EL],p[L_WR])
    J['R_elbow']=_angle(p[R_SH],p[R_EL],p[R_WR])
    J['L_knee']=_angle(p[L_HIP],p[L_KN],p[L_AN])
    J['R_knee']=_angle(p[R_HIP],p[R_KN],p[R_AN])
    J['L_hip']=_angle(p[L_SH],p[L_HIP],p[L_KN])
    J['R_hip']=_angle(p[R_SH],p[R_HIP],p[R_KN])
    return J

def compare_frames(ideal, user):
    i, u = _joint_angles(ideal[:,:3]), _joint_angles(user[:,:3])
    diffs = {k:u[k]-i[k] for k in i}
    mae = np.mean([abs(v) for v in diffs.values()])
    score = max(0, 100 - mae)
    return diffs, score

def analyze_sequences(ideal, user, mapping):
    per=[]
    for j,u in enumerate(user):
        diffs,score = compare_frames(ideal[mapping[j]],u)
        per.append({'diffs':diffs,'score':score})
    overall = np.mean([x['score'] for x in per])
    return per, overall
