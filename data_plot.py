import matplotlib.pyplot as plt

print("Which dataset?")
distribution = input()

file_name = "datasets/"
if distribution == "skewed" :
    file_name += distribution + "_1000000_4_2_.csv"
elif distribution == "japan":
    file_name += distribution + "_2030818_1_2_.csv"
elif distribution == "china":
    file_name += distribution + "_2677695_1_2_.csv"
else:
    file_name += distribution + "_1000000_1_2_.csv"

f = open(file_name)
x = []
y = []
cnt = 0
ex = 0
ey = 0

for row in f:
    #print(row)
    xi,yi,cnt = map(float,row.split(","))
    ex += xi
    ey += yi
    x.append(xi)
    y.append(yi)
    
'''
ex /= len(x)
ey /= len(y)
vx = vy = covxy = 0

for i in range(len(x)):
    vx += (ex - x[i])**2
    vy += (ey - y[i])**2
    covxy = (x[i] - ex) * (y[i] - ey)

vx /= len(x)
vy /= len(y)
covxy /= len(x)
print("ex :",ex,", ey :",ey)
print("vx :",vx,", vy :",vy)
print("covxy :",covxy)
exit()
'''

fig = plt.figure(dpi=200)
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,marker=".",s=0.05)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(0,1)
ax.set_xlim(0,1)
ax.set_aspect("equal")
fig.savefig(distribution+".png")