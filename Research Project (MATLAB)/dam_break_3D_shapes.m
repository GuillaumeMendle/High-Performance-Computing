%Set parameters

clear all;
close all;
clc;


%Set parameters

Nx=201; %number of points
s=0; %start of the domain
f=40; %end of the domain
dx=(f-s)/(Nx-1); %Delta x
ttotal=10000; %Total duration of the simulation
t=0; %Set t0=0;
g=9.81; %Gravity constant
x=s:dx:f; %Define the abscissa values
pausetime=0.1;
c=0.8;
D=0.06; %coefficient of artificial viscosity
count=0;

Ny=30/dx+1;

y=0:dx:30;



%------------------------------------------------

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

vv1=0;
vvo=0;
%----------------------------------------------


hhllc=zeros(Ny,Nx);

% for i=1:Ny
% for j=1:Nx
%     if x(j)<=xo
%         hhllc(i,j)=h1;
%     else
%         hhllc(i,j)=ho;
%     end
% end
% end

% hump

% for i=1:Nx
%     for j=1:Ny
%         hhllc(j,i)=max(0,0.8-0.01*((x(i)-20)^2+(y(j)-15)^2));
%     end
% end

% ellipse

% for i=1:Nx
%     for j=1:Ny
%         if ((x(i)-20)/10)^2+((y(j)-15)/1.5)^2<=1
%         hhllc(j,i)=0.2;
%         else 
%         hhllc(j,i)=0.0;
%         end
%     end
% end

%circle

% for i=1:Nx
%     for j=1:Ny
%         if (x(i)-20)^2+(y(j)-15)^2<=6.25
%         hhllc(j,i)=0.8;
%         else 
%         hhllc(j,i)=0.0;
%         end
%     end
% end

%cross

% for i=1:Nx
%     for j=1:Ny
%         if x(i)>=12.5 && x(i)<=27.5 && y(j)>=12.5 && y(j)<=17.5
%         hhllc(j,i)=0.8;
%         elseif x(i)>=17.5 && x(i)<=22.5 && y(j)>=7.5 && y(j)<=22.5
%         hhllc(j,i)=0.8;
%         else
%         hhllc(j,i)=0;
%         end
%     end
% end

%pill

for i=1:Nx
for j=1:Ny
   if (x(i)-10)^2+(y(j)-15)^2<=2.5
        hhllc(j,i)=0.8;
   elseif (x(i)-30)^2+(y(j)-15)^2<=2.5
        hhllc(j,i)=0.8;
   elseif x(i)>=10 & x(i)<=30
       if y(j)>=12.5 & y(j)<=17.5
        hhllc(j,i)=0.8;
       end
   end
       
end
end



uhllc=zeros(Ny,Nx);
for i=1:Ny
for j=1:Nx
    if x(j)<=xo
        uhllc(i,j)=uu1;
    else
        uhllc(i,j)=uuo;
    end
end
end

vhllc=zeros(Ny,Nx);

for i=1:Ny
for j=1:Nx
    if x(j)<=xo
        vhllc(i,j)=vv1;
    else
        vhllc(i,j)=vvo;
    end
end
end

zbin=zeros(Ny,Nx);
zb=zbin;

[hhllc,uhllc,vhllc]=boundaries(hhllc,uhllc,vhllc,Nx,Ny);

while t<ttotal
    
    neta=hhllc+zbin;
     
    netasave=neta;
    
    dt=deltat(dx,uhllc,vhllc,hhllc,g,c);
    lambda=dt/dx;

%----------------------------------------------------------------------

[un1,un2,un3,F1,F2,F3,G1,G2,G3]=variables(hhllc,uhllc,vhllc,neta,zb,g);



    for j=2:Ny-1
        
    for i=2:Nx-1
    
    
        F1L=F1(j,i);
        F2L=F2(j,i);
        F3L=F3(j,i);
        
        F1R=F1(j,i+1);
        F2R=F2(j,i+1);
        F3R=F3(j,i+1);

        
      
        [hstar,ustar]=star(hhllc,uhllc,g,i,j);
        [sL,sR]=vitesses(hhllc,uhllc,hstar,ustar,g,i,j);
        
        if sR-sL==0 
 
            F1star=0;
            F2star=0;
            F3star=0;
             
        else 
            F1star=(sR*F1(j,i)-sL*F1(j,i+1)+sL*sR*(un1(j,i+1)-un1(j,i)))/(sR-sL);
            F2star=(sR*F2(j,i)-sL*F2(j,i+1)+sL*sR*(un2(j,i+1)-un2(j,i)))/(sR-sL);
            F3star=(sR*F3(j,i)-sL*F3(j,i+1)+sL*sR*(un3(j,i+1)-un3(j,i)))/(sR-sL);
        end
        
    %-------------------------------------------------

        F1(j,i)=0.5*(F1L+F1R)-0.5*sL*lambda*(F1star-F1L)-0.5*sR*lambda*(F1R-F1star);
        F2(j,i)=0.5*(F2L+F2R)-0.5*sL*lambda*(F2star-F2L)-0.5*sR*lambda*(F2R-F2star);
        F3(j,i)=0.5*(F3L+F3R)-0.5*sL*lambda*(F3star-F3L)-0.5*sR*lambda*(F3R-F3star);

        
        
        un1(j,i)=real(un1(j,i)-lambda*(F1(j,i)-F1(j,i-1)));
        un2(j,i)=real(un2(j,i)-lambda*(F2(j,i)-F2(j,i-1)));
        un3(j,i)=real(un3(j,i)-lambda*(F3(j,i)-F3(j,i-1)));
        
        if hhllc(j,i+1)>0.000001

        un1(j,i)=un1(j,i)+D*(un1(j,i+1)-2*un1(j,i)+un1(j,i-1));
        un2(j,i)=un2(j,i)+D*(un2(j,i+1)-2*un2(j,i)+un2(j,i-1));
        un3(j,i)=un3(j,i)+D*(un3(j,i+1)-2*un3(j,i)+un3(j,i-1));
        end
    end
    end
    
%-------------------------------------------------------------------
[hhllc,uhllc,vhllc,neta]=variables2(un1,un2,un3,zb,Nx,Ny);

[hhllc,uhllc,vhllc]=boundaries(hhllc,uhllc,vhllc,Nx,Ny);
zb=zbin;
neta=hhllc+zbin;

[un1,un2,un3,F1,F2,F3,G1,G2,G3]=variables(hhllc,uhllc,vhllc,neta,zb,g);

%-----------------------------------------------------------------
  for j=2:Ny-1
    for i=2:Nx-1   

        G1L=G1(j,i);
        G2L=G2(j,i);
        G3L=G3(j,i);
        
        G1R=G1(j+1,i);
        G2R=G2(j+1,i);
        G3R=G3(j+1,i);
        
        [hstar,vstar]=star2(hhllc,vhllc,g,i,j);
        [sL,sR]=vitesses2(hhllc,vhllc,hstar,vstar,g,i,j);
        
           
        if sR-sL==0 
            G1star=0;
            G2star=0;
            G3star=0;
        else 
            G1star=(sR*G1(j,i)-sL*G1(j+1,i)+sL*sR*(un1(j+1,i)-un1(j,i)))/(sR-sL);
            G2star=(sR*G2(j,i)-sL*G2(j+1,i)+sL*sR*(un2(j+1,i)-un2(j,i)))/(sR-sL);
            G3star=(sR*G3(j,i)-sL*G3(j+1,i)+sL*sR*(un3(j+1,i)-un3(j,i)))/(sR-sL);
            

        end
        
        G1(j,i)=0.5*(G1L+G1R)-0.5*sL*lambda*(G1star-G1L)-0.5*sR*lambda*(G1R-G1star);
        G2(j,i)=0.5*(G2L+G2R)-0.5*sL*lambda*(G2star-G2L)-0.5*sR*lambda*(G2R-G2star);
        G3(j,i)=0.5*(G3L+G3R)-0.5*sL*lambda*(G3star-G3L)-0.5*sR*lambda*(G3R-G3star);

        

        un1(j,i)=real(un1(j,i)-lambda*(G1(j,i)-G1(j-1,i)));
        un2(j,i)=real(un2(j,i)-lambda*(G2(j,i)-G2(j-1,i)));
        un3(j,i)=real(un3(j,i)-lambda*(G3(j,i)-G3(j-1,i)));
        
        
       if hhllc(j+1,i)>0.000001

        un1(j,i)=un1(j,i)+D*(un1(j+1,i)-2*un1(j,i)+un1(j-1,i));
        un2(j,i)=un2(j,i)+D*(un2(j+1,i)-2*un2(j,i)+un2(j-1,i));
        un3(j,i)=un3(j,i)+D*(un3(j+1,i)-2*un3(j,i)+un3(j-1,i));
        
       end
         
    end
  end


[hhllc,uhllc,vhllc,neta]=variables2(un1,un2,un3,zb,Nx,Ny);

[hhllc,uhllc,vhllc]=boundaries(hhllc,uhllc,vhllc,Nx,Ny);

zb=zbin;
neta=hhllc+zbin;


mesh(x,y,neta)
shading flat %Remove the meshgrid lines
% colorbar

axis([0 40 0 30])
view(0,90)

pause(pausetime);     
      
     t=t+dt
    
end

        
    
%------------------------------------------------------------------------


function [u1,u2,u3,F1,F2,F3,G1,G2,G3]=variables(h,u,v,neta,zb,g)

u1=neta;
u2=h.*u;
u3=h.*v;

F1=h.*u;
F2=h.*u.^2+g.*(neta.^2-2*neta.*zb)/2;
F3=u.*v.*h;

G1=h.*v;
G2=u.*v.*h;
G3=h.*v.^2+g.*(neta.^2-2*neta.*zb)/2;


end


function [hhllc,uhllc,vhllc,neta]=variables2(U1,U2,U3,zb,Nx,Ny)

neta=U1;
hhllc=neta-zb;
uhllc=U2./hhllc;
vhllc=U3./hhllc;

for i=1:Nx
    for j=1:Ny
    if hhllc(j,i)==0
        uhllc(j,i)=0;
    end
    end
end

for i=1:Nx
    for j=1:Ny
    if hhllc(j,i)==0
        vhllc(j,i)=0;
    end
    end
end


end


function dt=deltat(dx,u,v,h,g,c)

maxi=max(max(sqrt(u.^2+v.^2)+sqrt(g*abs(h))));
dt=c*dx/maxi;
% dt=real(dt);

end


%--------------------------------------------------------------

function [hstar,ustar]=star(h,u,g,i,j)
cL=sqrt(h(j,i)*g);
cR=sqrt(h(j,i+1)*g);

ustar=(u(j,i)+u(j,i+1))/2+(cL-cR);
hstar=(((cL+cR)/2+(u(j,i)-u(j,i+1))/4)^2)/g;

end

function [hstar,vstar]=star2(h,v,g,i,j)
cL=sqrt(h(j,i)*g);
cR=sqrt(h(j+1,i)*g);

vstar=(v(j,i)+v(j+1,i))/2+(cL-cR);
hstar=(((cL+cR)/2+(v(j,i)-v(j+1,i))/4)^2)/g;

end

%-----------------------------------------------------------


function [sL,sR]=vitesses(h,u,hstar,ustar,g,i,j)
hL=h(j,i);
hR=h(j,i+1);
uL=u(j,i);
uR=u(j,i+1);
cL=sqrt(h(j,i)*g);
cR=sqrt(h(j,i+1)*g);

if hL==0
    sL=uR-2*cR;
else
    sL=min(uL-cL,ustar-sqrt(g*hstar));
end

if hR==0
    sR=uL+2*cL;
else
    sR=max(uR+cR,ustar+sqrt(g*hstar));
end

end

function [sL,sR]=vitesses2(h,v,hstar,vstar,g,i,j)
hL=h(j,i);
hR=h(j+1,i);
vL=v(j,i);
vR=v(j+1,i);
cL=sqrt(h(j,i)*g);
cR=sqrt(h(j+1,i)*g);

if hL==0
    sL=vR-2*cR;
else
    sL=min(vL-cL,vstar-sqrt(g*hstar));
end

if hR==0
    sR=vL+2*cL;
else
    sR=max(vR+cR,vstar+sqrt(g*hstar));
end

end


%--------------------------------------------------------------


function sstar=vitessesstar(h,u,sL,sR,g,i)
    
hL=h(i);
hR=h(i+1);
uL=u(i);
uR=u(i+1);
cL=sqrt(h(i)*g);
cR=sqrt(h(i+1)*g);

sstar=(sL*hR*(uR-sR)-sR*hL*(uL-sL))/(hR*(uR-sR)-hL*(uL-sL));

end

function [h,u,v]=boundaries(h,u,v,Nx,Ny)

for i=1:Nx
    h(1,i)=h(2,i);
    h(Ny,i)=h(Ny-1,i);
    u(1,i)=0;
%     u(1,i)=u(2,i);
    u(Ny,i)=0;
%     u(Ny,i)=u(Ny-1,i);

end

for j=1:Ny
    h(j,Nx)=h(j,Nx-1);
    h(j,1)=h(j,2);
    u(j,Nx)=0;
    u(j,1)=0;
end
end


%----MINBEE LIMITER----


function [phi1,phi2]=phiMinbee(hhllc,Cone,Ctwo,i,j)


if Cone>0
    
Rone=(hhllc(j,i)-hhllc(j,i-1))/(hhllc(j,i+1)-hhllc(j,i));

elseif Cone<0
    
Rone=(hhllc(j,i+2)-hhllc(j,i+1))/(hhllc(j,i+1)-hhllc(j,i));

else
    Rone=2;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(j,i+1)-hhllc(j,i))/(hhllc(j,i+2)-hhllc(j,i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(j,i+3)-hhllc(j,i+2))/(hhllc(j,i+2)-hhllc(j,i+1));
        
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


function [phi1,phi2]=phiSuperbee(hhllc,Cone,Ctwo,i,j)


if Cone>0
    
Rone=(hhllc(j,i)-hhllc(j,i-1))/(hhllc(j,i+1)-hhllc(j,i));

elseif Cone<0
    
Rone=(hhllc(j,i+2)-hhllc(j,i+1))/(hhllc(j,i+1)-hhllc(j,i));

else
    Rone=-1;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(j,i+1)-hhllc(j,i))/(hhllc(j,i+2)-hhllc(j,i+1));
        
elseif Ctwo<0
        Rtwo=(hhllc(j,i+3)-hhllc(j,i+2))/(hhllc(j,i+2)-hhllc(j,i+1));
        
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

%---------------------------------------------------------------------

%----MINBEE LIMITER 2----


function [phi1,phi2]=phiMinbee22(hhllc,Cone,Ctwo,i,j)


if Cone>0
    
Rone=(hhllc(j,i)-hhllc(j-1,i))/(hhllc(j+1,i)-hhllc(j,i));

elseif Cone<0
    
Rone=(hhllc(j+2,i)-hhllc(j+1,i))/(hhllc(j+1,i)-hhllc(j,i));

else
    Rone=2;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(j+1,i)-hhllc(j,i))/(hhllc(j+2,i)-hhllc(j+1,i));
        
elseif Ctwo<0
        Rtwo=(hhllc(j+3,i)-hhllc(j+2,i))/(hhllc(j+2,i)-hhllc(j+1,i));
        
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




%----SUPERBEE LIMITER 2----


function [phi1,phi2]=phiSuperbee2(hhllc,Cone,Ctwo,i,j)


if Cone>0
    
Rone=(hhllc(j,i)-hhllc(j-1,i))/(hhllc(j+1,i)-hhllc(j,i));

elseif Cone<0
    
Rone=(hhllc(j+2,i)-hhllc(j+1,i))/(hhllc(j+1,i)-hhllc(j,i));

else
    Rone=-1;

end

    
if Ctwo>0
    
        Rtwo=(hhllc(j+1,i)-hhllc(j,i))/(hhllc(j+2,i)-hhllc(j+1,i));
        
elseif Ctwo<0
        Rtwo=(hhllc(j+3,i)-hhllc(j+2,i))/(hhllc(j+2,i)-hhllc(j+1,i));
        
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
