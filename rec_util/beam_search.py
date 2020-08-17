import numpy as np

def beam_search(probs, beam_size=5):
	'''
	:param probs:  bs,L,N
	:param beam_size:
	:return:
	'''
	B, L, N  = probs.shape
	probs = np.log(probs)
	final_ans = []
	for i in range(B):
		probs_b = probs[i]
		ans_b = [([], 0.)]
		for j in range(L):
			cur_ans_b = []
			for item in ans_b:
				for v in N:
					new_str = item[0] + [chars[v]]
					new_score = item[1] + probs_b[j, v]
					cur_ans_b.append((new_str, new_score))
			cur_ans_b = sorted(cur_ans_b, key=lambda x: x[1], reverse=True)
			ans_b = cur_ans_b[:beam_size]
			final_ans.append(ans_b)
	return final_ans