import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("Which dataset?")
distribution = input()

file_name = "datasets/"
if distribution == "skewed" :
    file_name += distribution + "_1000000_4_2_.csv"
elif distribution == "japan":
    file_name += distribution + "_2030818_1_2_.csv"
elif distribution == "china":
    file_name += distribution + "_2677695_1_2_.csv"
elif distribution == "usa":
    file_name += distribution + "_17383488_1_2_.csv"
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
fig = plt.figure(dpi=400)
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,marker=".",s=0.05)

ans = input("plot Rectangle? (y/n)\n")

if ans == "y":
    f = open(distribution + "_mbr.out")
    for row in f:
        l = row.split()
        level = int(l[2])
        if level == 0:
            continue
        elif level == 1:
            color = "#000000"
        elif level == 2:
            color = "#00ff00"
        elif level == 3:
            color = "#00ffff"
        else:
            color = "#ff0000"
        x1 = float(l[7])
        y1 = float(l[9])
        x2 = float(l[13])
        y2 = float(l[15])
        lw = 0.5 / level
        width = x2 - x1
        height = y2 -y1
        r = patches.Rectangle(xy=(x1, y1), width = width, height = height, ec=color, linewidth = str(lw) ,fill=False)
        ax.add_patch(r)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylim(0,1)
ax.set_xlim(0,1)
ax.set_aspect("equal")
fig.savefig(distribution+".png")