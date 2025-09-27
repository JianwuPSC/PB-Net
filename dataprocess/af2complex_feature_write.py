n=0
txt=[]
f=open('/home/wuj/data/tools/af2complex/example/1.sh')
for line in f:
    txt.append(line.strip())        
    if n % 12 ==0:
        txt.append('wait')
    n=n+1


f=open("/home/wuj/data/tools/af2complex/example/zz.sh","w")

for line in txt:
    f.write(line+'\n')
f.close()
