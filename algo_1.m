%% Load Input

m=10; %number of circles 
n = 100; %number of points for in given circle
s=5;% number of segnments that we will initially cluster
r=50; %radius is fixed for all circles, my be changed
angles = 2*pi*rand(1,n); %anglular poistion of a point in a circle

for i=1:m
  cx= rand*100;  %a random x center coordinate
  cy= rand*100;  %a random y center coordinate
x=r*cos(angles)+cx;  %parametric equation of a circle
y=r*sin(angles)+cy;
c=(randi([1, s]))/10; 
color=c.*[0.25,0.5,0];
scatter(x,y,'MarkerFaceColor',color);
hold on;
end


%% Set uniform column sampling
% Initialise clusters




%% While loop 
