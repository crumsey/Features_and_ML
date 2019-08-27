import matplotlib as mpl
mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size":18, "figure.figsize":(10,8)})
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def mycontour(figname, x, y, z, colormap="RdBu_r", lines=False):
    
    plt.figure(figname)
    plt.tricontour(x ,y ,z, levels=30, linewidths=0.5, colors='k')
    plt.tricontourf(x, y, z, levels=30, cmap=colormap)


def myplot(figname, x, y, style, linewidth, label):

    plt.figure(figname)
    plt.plot(x, y, style, linewidth=linewidth, label=label)

def mysemilogx(figname, x, y, style, linewidth, label):

    plt.figure(figname)
    plt.semilogx(x, y, style, linewidth=linewidth, label=label)

def mysemilogy(figname, x, y, style, linewidth, label):

    plt.figure(figname)
    plt.semilogy(x, y, style, linewidth=linewidth, label=label)

def myscatter(figname, x, y, style, label):

    plt.figure(figname)
    plt.scatter(x, y, style, label=label)

def myfig(figname, xlabel, ylabel, title, legend=False):

    plt.figure(figname)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if legend==True:
        plt.legend()

def myfigshow():

    plt.show()

def myfigsave(foldername, figname):
    
    plt.figure(figname)
    plt.savefig("%s/figs/%s"%(foldername, figname))

if __name__=="__main__":
        
        import numpy as np
        x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.])
        myplot(1, x, x**2, '-g', 2, None)
        myfigshow(1)
