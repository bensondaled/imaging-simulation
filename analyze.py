from structured_nmf import sNMF
from sklearn.utils.linear_assignment_ import linear_assignment
import tifffile

def normalize(a, axis):
    a = (a-a.min(axis=axis))/(a.max(axis=axis)-a.min(axis=axis))
    return a/a.sum(axis=axis)

path = '/jukebox/wang/deverett/simulations/batch_9/01_018/01_018'
mov = tifffile.imread(path+'.tif')

snmf = sNMF(mov)
snmf.run()

with np.load(path+'.npz') as data:
    input_masks = np.array([c['mask_im'].flatten() for c in data['cells'] if c['was_in_fov']]).T # n_pixels x n_masks
output_masks = np.asarray(snmf.A.todense()) # n_pixels x n_masks
output_masks = np.array([m.reshape([128,64]).T.flatten() for m in output_masks.T]).T 
powers = (output_masks.sum(axis=0)/((output_masks!=0).sum(axis=0)))**2
output_masks = normalize(output_masks, axis=0)
output_masks = output_masks.T # now n_masks x n_pixels

similarity = np.dot(output_masks, input_masks)
cost = similarity.max() - similarity
assignment = linear_assignment(cost)
match_quality = np.array([similarity[a[0],a[1]] for a in assignment])

# show that the coordinates of matches line up nicely:
for a,mc in zip(assignment,match_quality): 
    pl.scatter(np.mean(np.argwhere(output_masks[a[0]])), np.mean(np.argwhere(input_masks[:,a[1]])), c=['r','g'][int(mc>0.05)])   

i = 52
om = (output_masks[assignment[i,0]].reshape([64,128])>0).astype(int)
im = (input_masks[:,assignment[i,1]].reshape([64,128])>0).astype(int)
pl.imshow(im+om)

# TODO
# find some quality of output_masks, such that pl.scatter(match_quality, X[assignment[:,0]]), where X lines up with output_masks, is a nice relationship. so it can be used as a determinant of quality in the absence of ground truth
