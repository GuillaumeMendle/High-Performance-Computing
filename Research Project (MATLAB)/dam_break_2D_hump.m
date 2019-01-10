
%Set parameters

clear all;
close all;
clc;

%Set parameters

N=501; %number of points
s=0; %start of the domain
f=40; %end of the domain
dx=(f-s)/(N-1); %Delta x
ttotal=100; %Total duration of the simulation
t=0; %Set t0=0;
g=9.81; %Gravity constant
x=s:dx:f; %Define the abscissa values
pausetime=0.1;
c=0.8;
D=0.06; %coefficient of artificial viscosity

u=zeros(1,N); %Define u



%Define initial conditions

h1=0.75;
ho=0.0;

if ho==0
    Limiter=1;
else 
    Limiter=3;
end

uu1=0;
uuo=0;
xo=15.5;


%-------------------------------------------------------


zb=zeros(1,N);

a=0.4/3;

for i=1:N
    if x(i)<=23.6
        zbin(i)=max(0,a*(x(i)-20.6));
    end
    
    if x(i)>=23.6
        zbin(i)=max(0,0.4-a*(x(i)-23.6));
    end
end


zb=zbin;

%--------------------------


for i=1:N
    if x(i)<=xo
        hhllc(i)=h1;
    else
        hhllc(i)=ho;
    end
end

for i=1:N
    if x(i)<=xo
        u(i)=uu1;
    else
        u(i)=uuo;
    end
end


%Define initial conditions


%---------------------------------------------------------



[hhllc, u]=boundaries(hhllc,u,N);

while t<ttotal
    
    neta=hhllc+zbin;
    
    netasave=neta;
    
    [hhllc,u,neta]=boundaries2(hhllc,u,neta,zb,x,t,N);
    

    
    dt=deltat(dx,u,hhllc,g,c);
    lambda=dt/dx;

    
    %----------------------------------------------------------

for i=2:N-1
    
if neta(i)<neta(i+1) && u(i+1)==0 && zbin(i+1)>zbin(i)
    
deltaneta=neta(i+1)-neta(i);
neta(i+1)=neta(i+1)-deltaneta;
zb(i+1)=neta(i+1);
u(i)=0;
end
end

for i=2:N-1
    
if neta(i)>neta(i+1) && u(i+1)==0 && zbin(i+1)<zbin(i)
u(i)=0;
end
end

[un1,un2,F1,F2]=variables(hhllc,u,neta,zb,g);


    for i=2:N-1
        
        
        
        F1L=F1(i);
        F2L=F2(i);
        F1R=F1(i+1);
        F2R=F2(i+1);
        
        [hstar,ustar]=star(hhllc,u,g,i);
        [sL,sR]=vitesses(hhllc,u,hstar,ustar,g,i);
        
        if sR-sL==0 

            F1star=0;
            F2star=0; 

        else 
            F1star=(sR*F1(i)-sL*F1(i+1)+sL*sR*(un1(i+1)-un1(i)))/(sR-sL);
            F2star=(sR*F2(i)-sL*F2(i+1)+sL*sR*(un2(i+1)-un2(i)))/(sR-sL);
        end
       
        
%-------------------------------------------------

        if i>=2 && i<=N-3
           
        
        Cone=sL*lambda;
        Ctwo=sR*lambda;
        
        if Limiter==1
        [phi1,phi2]=phiMinbee(hhllc,Cone,Ctwo,i);
        elseif Limiter==2
            [phi1,phi2]=phiMinbee2(hhllc,Cone,Ctwo,i);
        elseif Limiter==3
            [phi1,phi2]=phiSuperbee(hhllc,Cone,Ctwo,i);
        elseif Limiter==4
            [phi1,phi2]=phiLeer(hhllc,Cone,Ctwo,i);
        elseif Limiter==5
            [phi1,phi2]=phiAlbada(hhllc,Cone,Ctwo,i);
        end
        
 
    
        F1(i)=0.5*(F1L+F1R)-0.5*sign(Cone)*phi1*(F1star-F1L)-0.5*sign(Ctwo)*phi2*(F1R-F1star);
        F2(i)=0.5*(F2L+F2R)-0.5*sign(Cone)*phi1*(F2star-F2L)-0.5*sign(Ctwo)*phi2*(F2R-F2star);
        
        
        un1(i)=real(un1(i)-lambda*(F1(i)-F1(i-1)));
        un2(i)=real(un2(i)-lambda*(F2(i)-F2(i-1))-g*neta(i)*lambda*(zb(i+1)-zb(i)));

        if hhllc(i+1)>0.000001
            
        un1(i)=un1(i)+D*(un1(i+1)-2*un1(i)+un1(i-1));
        un2(i)=un2(i)+D*(un2(i+1)-2*un2(i)+un2(i-1));
        end
        
        else
        F1(i)=0.5*(F1L+F1R)-0.5*sL*lambda*(F1star-F1L)-0.5*sR*lambda*(F1R-F1star);
        F2(i)=0.5*(F2L+F2R)-0.5*sL*lambda*(F2star-F2L)-0.5*sR*lambda*(F2R-F2star);
        
        un1(i)=real(un1(i)-lambda*(F1(i)-F1(i-1)));
        un2(i)=real(un2(i)-lambda*(F2(i)-F2(i-1))-g*neta(i)*lambda*(zb(i+1)-zb(i)));
        

              if hhllc(i+1)>0.000001
        un1(i)=un1(i)+D*(un1(i+1)-2*un1(i)+un1(i-1));
        un2(i)=un2(i)+D*(un2(i+1)-2*un2(i)+un2(i-1));

              end

        end
end  
 
       
    [hhllc,u,neta]=variables2(un1,un2,zb,N);
    [hhllc, u]=boundaries(hhllc,u,N);

   


    zb=zbin;
    neta=hhllc+zbin;
   
    [hhllc,u,neta]=boundaries2(hhllc,u,neta,zbin,x,t,N);

    plot(x,neta,'b',x,zbin,'g')
    
    axis([s f 0 1.1])
    
    pause(pausetime); 
    
    t=t+dt
end   
    
%------------------------------------------------------------------------


function[u1,u2,F1,F2]=variables(h,u,neta,zb,g)

u1=neta;
u2=h.*u;

F1=h.*u;
F2=h.*u.^2+g.*(neta.^2-2*neta.*zb)/2;

end


function [h,u,neta]=variables2(U1,U2,zb,N)

neta=U1;
h=neta-zb;
u=U2./h;


for i=1:N
    if h(i)==0
        u(i)=0;
    end
end


end

function dt=deltat(dx,u,h,g,c)

maxi=max(abs(u)+sqrt(abs(g*h)));
dt=c*dx/maxi;

end


function [hstar,ustar]=star(h,u,g,i)
cL=sqrt(h(i)*g);
cR=sqrt(h(i+1)*g);

ustar=(u(i)+u(i+1))/2+(cL-cR);
hstar=(((cL+cR)/2+(u(i)-u(i+1))/4)^2)/g;

end

function [sL,sR]=vitesses(h,u,hstar,ustar,g,i)
hL=h(i);
hR=h(i+1);
uL=u(i);
uR=u(i+1);
cL=sqrt(h(i)*g);
cR=sqrt(h(i+1)*g);

if hL<0.000001
    sL=uR-2*cR;
else
    sL=min(uL-cL,ustar-sqrt(g*hstar));
end

if hR<0.000001
    sR=uL+2*cL;
else
    sR=max(uR+cR,ustar+sqrt(g*hstar));
end

end



function sstar=vitessesstar(h,u,sL,sR,g,i)
    
hL=h(i);
hR=h(i+1);
uL=u(i);
uR=u(i+1);
cL=sqrt(h(i)*g);
cR=sqrt(h(i+1)*g);

sstar=(sL*hR*(uR-sR)-sR*hL*(uL-sL))/(hR*(uR-sR)-hL*(uL-sL));

end

function [h,u]=boundaries(h,u,N)

h(1)=h(2);

u(1)=0;
u(2)=0;

h(N)=h(N-1);

u(N)=u(N-1);


end


% function [hhllc,uhllc]=boundaries2(hhllc,uhllc,neta,zb,N)
% 
% for i=1:N
% if hhllc(i)<=abs(zb(i)+0.001) && zb(i)>0
%     uhllc(i)=0;
% end
% 
% if neta(i)<=zb(i) 
%    neta(i)=zb(i);
% end
% 
% end
% 
% 
% end

function [hhllc,uhllc,neta]=boundaries2(hhllc,uhllc,neta,zb,x,t,N)
for i=1:N
    if x(i)>=23.6 && x(i)<=26.6 && abs(neta(i)-zb(i))<0.001 && t>10
        neta(i)=zb(i);
        hhllc(i)=0;
%         deltaneta=neta(i)-neta(i+1);
%         neta(i)=neta(i)-deltaneta;
%         zb(i)=neta(i);
    end
end
end





%----MINBEE LIMITER----


function [phi1,phi2]=phiMinbee(hhllc,Cone,Ctwo,i)


if Cone>0
    
Rone=(hhllc(i)-hhllc(i-1))/(hhllc(i+1)-hhllc(i));

elseif Cone<0
    
Rone=(hhllc(i+2)-hhllc(i+1))/(hhllc(i+1)-hhllc(i));

else
    Rone=2;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(i+1)-hhllc(i))/(hhllc(i+2)-hhllc(i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(i+3)-hhllc(i+2))/(hhllc(i+2)-hhllc(i+1));
        
else 
    Rtwo=2;
        
end


if Rone<0
    phi1=1;
elseif 0<=Rone && Rone<=1
    phi1=Rone;
else
    phi1=abs(Cone);
end

if Rtwo<=0
    
    phi2=1;
    
elseif 0<Rtwo && Rtwo<=1
    
    phi2=Rtwo;
else
    phi2=abs(Ctwo);
end

end


%----MINBEE LIMITER MODIFIED----




function [phi1,phi2]=phiMinbee2(hhllc,Cone,Ctwo,i)


if Cone>0
    
Rone=(hhllc(i)-hhllc(i-1))/(hhllc(i+1)-hhllc(i));

else
    
Rone=(hhllc(i+2)-hhllc(i+1))/(hhllc(i+1)-hhllc(i));
end

if Ctwo>0
    
        Rtwo=(hhllc(i+1)-hhllc(i))/(hhllc(i+2)-hhllc(i+1));
        
else
        Rtwo=(hhllc(i+3)-hhllc(i+2))/(hhllc(i+2)-hhllc(i+1));
        
end

if Rone<=0
    phi1=1;
elseif 0<Rone && Rone<=1
    phi1=Rone;
else
    phi1=abs(Cone);
end

if Rone<=0
    
    phi2=1;
    
elseif 0<Rone && Rone<=1
    
    phi2=Rone;
else
    phi2=abs(Ctwo);
end

end



%----SUPERBEE LIMITER----


function [phi1,phi2]=phiSuperbee(hhllc,Cone,Ctwo,i)


if Cone>0
    
Rone=(hhllc(i)-hhllc(i-1))/(hhllc(i+1)-hhllc(i));

elseif Cone<0
    
Rone=(hhllc(i+2)-hhllc(i+1))/(hhllc(i+1)-hhllc(i));

else
    Rone=-1;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(i+1)-hhllc(i))/(hhllc(i+2)-hhllc(i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(i+3)-hhllc(i+2))/(hhllc(i+2)-hhllc(i+1));
        
else 
    Rtwo=-1;
        
end


if Rone<=0
    phi1=1;
elseif 0<Rone && Rone<=1/2
    phi1=1-2*(1-abs(Cone))*Rone;
elseif 1/2<Rone<=1
    phi1=abs(Cone);
    
    elseif 1<Rone && Rone<=2
       phi1=1-(1-abs(Cone))*Rone;
else
    phi1=2*abs(Cone)-1;
    
end

if Rtwo<=0
    phi2=1;
elseif 0<Rtwo && Rtwo<=1/2
    phi2=1-2*(1-abs(Ctwo))*Rtwo;
elseif 1/2<Rtwo<=1
    phi2=abs(Ctwo);
    
    elseif 1<Rtwo && Rtwo<=2
       phi2=1-(1-abs(Ctwo))*Rtwo;
else
    phi2=2*abs(Ctwo)-1;
    
end


end

%----VAN LEER'S LIMITER----

function [phi1,phi2]=phiLeer(hhllc,Cone,Ctwo,i)


if Cone>0
    
Rone=(hhllc(i)-hhllc(i-1))/(hhllc(i+1)-hhllc(i));

elseif Cone<0
    
Rone=(hhllc(i+2)-hhllc(i+1))/(hhllc(i+1)-hhllc(i));

else
    Rone=-1;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(i+1)-hhllc(i))/(hhllc(i+2)-hhllc(i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(i+3)-hhllc(i+2))/(hhllc(i+2)-hhllc(i+1));
        
else 
    Rtwo=-1;
        
end


if Rone<=0
    phi1=1;
else
    phi1=1-(1-abs(Cone))*2*Rone/(1+Rone);
end

if Rtwo<=0
    phi2=1;
else
    phi2=1-(1-abs(Ctwo))*2*Rtwo/(1+Rtwo);
end

end



%----VAN ALBADA'S LIMITER----


function [phi1,phi2]=phiAlbada(hhllc,Cone,Ctwo,i)


if Cone>0
    
Rone=(hhllc(i)-hhllc(i-1))/(hhllc(i+1)-hhllc(i));

elseif Cone<0
    
Rone=(hhllc(i+2)-hhllc(i+1))/(hhllc(i+1)-hhllc(i));

else
    Rone=-1;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(i+1)-hhllc(i))/(hhllc(i+2)-hhllc(i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(i+3)-hhllc(i+2))/(hhllc(i+2)-hhllc(i+1));
        
else 
    Rtwo=-1;
        
end


if Rone<=0
    phi1=1;
else
    phi1=1-(1-abs(Cone))*Rone*(1+abs(Rone))/(1+Rone^2);
end

if Rtwo<=0
    phi2=1;
else
    phi2=1-(1-abs(Ctwo))*Rtwo*(1+abs(Rtwo))/(1+Rtwo^2);
end

end

