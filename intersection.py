

def intersection(a, b):
    output=[]
    for i in range(0,a.shape[0]):
        for j in range(0,b.shape[0]):
            if a[i]==b[j]:
                output.append(a[i])

    return output