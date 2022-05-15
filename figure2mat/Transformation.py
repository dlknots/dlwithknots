
from figure2mat import f2m



k=5
for i in range(1,k+1):
   pth="pics/k"+str(i)+".png"
   a=f2m(pth)
   
   sav="01to02/"+"k"+str(i)
   a.f2mat(pth,matname=sav,savename="b")

# pth="pics/k6"+".png"
# a=f2m(pth)
# sav="01to02/"+"k6"