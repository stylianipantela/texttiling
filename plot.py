from numpy import genfromtxt
import matplotlib.pyplot as plt

# set some nicer defaults for matplotlib
from matplotlib import rcParams

#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 100
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Thank you CS109 for the great function: http://nbviewer.ipython.org/github/cs109/content/blob/master/HW3_solutions.ipynb
    Minimize chartjunk by stripping out unnecesary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def plot_ws(filename):
  f = genfromtxt(filename + '.csv', names=True, delimiter=',')
  remove_border()
  
  plt.plot(f['w'], f['precision'], 'k')
  plt.plot(f['w'], f['recall'], 'b')
  plt.plot(f['w'], f['window_difference'], 'r')

  plt.ylabel('Accuracy Score', fontdict={'fontsize':10})
  plt.xlabel('W', fontdict={'fontsize':10})
  plt.title('Varying size of W', fontdict={'fontsize':20})
  plt.legend(['Precision', 'Recall', 'Window Difference'], loc='best', frameon=True)
  plt.savefig(filename + '.png')
  plt.show()

def plot_ks(filename):
  f = genfromtxt(filename + '.csv', names=True, delimiter=',')
  remove_border()
  
  plt.plot(f['k'], f['precision'], 'k')
  plt.plot(f['k'], f['recall'], 'b')
  plt.plot(f['k'], f['window_difference'], 'r')

  plt.ylabel('Accuracy Score', fontdict={'fontsize':10})
  plt.xlabel('K', fontdict={'fontsize':10})
  plt.title('Varying size of K', fontdict={'fontsize':20})
  plt.legend(['Precision', 'Recall', 'Window Difference'], loc='best', frameon=True)
  plt.savefig(filename + '.png')
  plt.show()


