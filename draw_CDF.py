import matplotlib.pyplot as plt
import math
print("Which dataset?")
distribution = input()

file_name = "datasets/"
if distribution == "skewed" :
    file_name += distribution + "_1000000_4_2_.csv"
    n = 1000000
elif distribution == "japan":
    file_name += distribution + "_2030818_1_2_.csv"
    n = 2030818
elif distribution == "china":
    file_name += distribution + "_2677695_1_2_.csv"
    n = 2677695
elif distribution == "usa":
    file_name += distribution + "_17383488_1_2_.csv"
    n = 17383488
else:
    file_name += distribution + "_1000000_1_2_.csv"
    n = 1000000

f = open(file_name)
x = []
y = []
bit_num = math.ceil(math.log(n,2))

def calc_Z(x,y):
    result = 0
    for i in range(bit_num):
        seed = pow(2,i)
        tmp = seed & x
        tmp = tmp << i
        result += tmp
        tmp = seed & y
        tmp = tmp << (i+1)
        result += tmp
    return result

def my_calc_Z(x,y):
    z = 0
    for i in range(bit_num):
        z += pow(2,i*2+1)*(y >> i & 1)
        z += pow(2,i*2)*(x >> i & 1)
    return z

for row in f:
    #print(row)
    xi,yi,cnt = map(float,row.split(","))
    xi = int(xi * n)
    yi = int(yi * n)
    z = calc_Z(xi,yi)
    x.append(z)
    y.append(cnt + 1)
x.sort()

fig = plt.figure(dpi=200)
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,marker=".",s=0.1)

ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_ylim(0,1)
#ax.set_xlim(0,1)
#ax.set_aspect("equal")
fig.savefig(distribution + "_CDF.png")