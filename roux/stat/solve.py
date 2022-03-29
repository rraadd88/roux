import numpy as np

def get_intersection_locations(y1: np.array,y2: np.array,test: bool=False,x: np.array=None) -> list: 
    """Get co-ordinates of the intersection (x[idx]).

    Args:
        y1 (np.array): vector.
        y2 (np.array): vector.
        test (bool, optional): test mode. Defaults to False.
        x (np.array, optional): vector. Defaults to None.

    Returns:
        list: output.
    """
    idxs=np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    if test:
        x=range(len(y1)) if x is None else x
        plt.figure(figsize=[2.5,2.5])
        ax=plt.subplot()
        ax.plot(x,y1,color='r',label='line1',alpha=0.5)
        ax.plot(x,y2,color='b',label='line2',alpha=0.5)
        _=[ax.axvline(x[i],color='k') for i in idxs]
        _=[ax.text(x[i],ax.get_ylim()[1],f"{x[i]:1.1f}",ha='center',va='bottom') for i in idxs]
        ax.legend(bbox_to_anchor=[1,1])
        ax.set(xlabel='x',ylabel='density')
    return idxs

# def get_intersection_of_gaussians(m1, s1, m2, s2):
#     # coefficients of quadratic equation ax^2 + bx + c = 0
#     a = (s1**2.0) - (s2**2.0)
#     b = 2 * (m1 * s2**2.0 - m2 * s1**2.0)
#     c = m2**2.0 * s1**2.0 - m1**2.0 * s2**2.0 - 2 * s1**2.0 * s2**2.0 * np.log(s1/s2)
#     x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
#     x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
#     return x1, x2

# def get_intersection_of_gaussians(m1,m2,std1,std2):
#     a = 1/(2*std1**2) - 1/(2*std2**2)
#     b = m2/(std2**2) - m1/(std1**2)
#     c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
#     return np.roots([a,b,c])