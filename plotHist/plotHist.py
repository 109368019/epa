import numpy as np 
import matplotlib.pyplot as plt 
import pandas

if(__name__ == "__main__"):
    df = pandas.read_csv("ref_val_sat - 工作表2.csv", header=0)
    
    data = np.array(df)
    
    bin_count = 100
    xx = np.arange(bin_count+1)
    x = (xx+0.5)[0:-1]
    xx = xx/bin_count
    x = x/bin_count

    fig = plt.figure()
    ref_v = data[:, 0]
    ref_v = ref_v[ref_v==ref_v]
    y, bin_ = np.histogram(ref_v, bins=xx)
    y = y/y.sum()
    plt.plot(x, y, "-", label="Refrigerant Value")

    all_v = data[:, 2]
    all_v = all_v[all_v==all_v]
    y, bin_ = np.histogram(all_v, bins=xx)
    y = y/y.sum()
    plt.plot(x, y, "--", label="Other Object Value")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Intensity")
    plt.ylabel("Probability(%)")
    plt.title("Value Histogram")

    fig.tight_layout()
    plt.savefig("valueHist.png")

    #=======================================
    xx = np.arange(bin_count/2+1)
    x = (xx+0.5)[0:-1]
    xx = xx/bin_count
    x = x/bin_count

    fig = plt.figure()
    ref_v = data[:, 1]
    ref_v = ref_v[ref_v==ref_v]
    y, bin_ = np.histogram(ref_v, bins=xx)
    y = y/y.sum()
    plt.plot(x, y, "-", label="Refrigerant Saturation")

    all_v = data[:, 3]
    all_v = all_v[all_v==all_v]
    y, bin_ = np.histogram(all_v, bins=xx)
    y = y/y.sum()
    plt.plot(x, y, "--", label="Other Object Saturation")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Intensity")
    plt.ylabel("Probability(%)")
    plt.title("Saturation Histogram")

    fig.tight_layout()
    plt.savefig("satHist.png")

