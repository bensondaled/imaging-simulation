from structured_nmf import sNMF
from sklearn.utils.linear_assignment_ import linear_assignment
import tifffile
from matplotlib import colors

path = '/jukebox/wang/deverett/simulations/batch_9/01_005/01_005'
path = 'example_output/example_output'
mov = tifffile.imread(path+'.tif')

snmf = sNMF(mov)
snmf.run()

with np.load(path+'.npz') as data:
    input_masks = np.array([c['mask_im'].flatten() for c in data['cells'] if c['was_in_fov']]) # nmasksxnpixels
output_masks = np.asarray(snmf.A.todense()).T # nmasksxnpixels
output_masks = np.array([m.reshape([128,64]).T.flatten() for m in output_masks])
out_thresh = 0.07 #value in output mask that will be considered part of mask
output_masks[output_masks<out_thresh] = 0

similarity = np.corrcoef(output_masks, input_masks)
similarity = similarity[:output_masks.shape[0],-input_masks.shape[0]:]
similarity[similarity<0]=0 #negative correlations are as bad as 0
cost = 1.0 - similarity
assignment = linear_assignment(cost)
match_quality = np.array([similarity[a[0],a[1]] for a in assignment])

# single:
i = 0
om = (output_masks[assignment[i,0]].reshape([64,128])>0).astype(int)
im = (input_masks[assignment[i,1]].reshape([64,128])>0).astype(int)
pl.imshow(im+om)

# all:
quality_thresh = 0.5
idxs = np.argwhere(match_quality>quality_thresh).squeeze()
out = np.array([(output_masks[assignment[i,0]].reshape([64,128])>0).astype(int) for i in idxs]).sum(axis=0)
inn = np.array([(input_masks[assignment[i,1]].reshape([64,128])>0).astype(int) for i in idxs]).sum(axis=0)

toshow = 1*inn+2*out
toshow[toshow>3] = 3
cmap = colors.ListedColormap(['blue','red','purple'])
pl.imshow(np.ma.masked_where(toshow==0, toshow),cmap=cmap)
pl.title('blue=input, red=output, purple=overlap\n frac \'found\'=%0.2f'%(np.sum(match_quality>=quality_thresh)/float(cost.shape[1])))


# TODO
# find some quality of output_masks, such that pl.scatter(match_quality, X[assignment[:,0]]), where X lines up with output_masks, is a nice relationship. so it can be used as a determinant of quality in the absence of ground truth
# play with: gsiz, gsig, dist, minsize, maxsize (last 3 must be updated in multiple functions)

"""
# OLD METHOD:
def normalize(a, axis):
    a = (a-a.min(axis=axis))/(a.max(axis=axis)-a.min(axis=axis))
    return a/a.sum(axis=axis)

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

"""
