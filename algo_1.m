%% Load Input

num_circle=5; %number of circles 
num_points = 100; %number of points for in given circle
s=5;% number of segnments that we will initially cluster
r=50; %radius is fixed for all circles, my be changed
angles = 2*pi*rand(1,num_points); %anglular poistion of a point in a circle
%Dimension=2; % NUmber of dimension
data = [];
for i=1:num_circle
  cx= rand*100;  %a random x center coordinate
  cy= rand*100;  %a random y center coordinate
  x=r*cos(angles)+cx;  %parametric equation of a circle
  y=r*sin(angles)+cy;
  c=(randi([1, s]))/10; 
  color=c.*[0.25,0.5,0];
  temp = [x;y];
  data = [data, temp];
  scatter(x,y,'MarkerFaceColor',color);
  hold on;
end


clearvars x y temp cx cy color c angles r s i;

data =data'; % This is my data matrix

%% Set uniform column sampling
% Initialise clusters
%N = m*num_points;
N=500;
K = 5; % Number of clusters
cluster = zeros(N,K); % Cluster matrix

%% While loop 
n = 5; % Number of points to build model
error = 1000;
T =100; % number of column of P matrix (1<= T <= Nc)
while error>0.1
    for t = 1:T
        r =randi([1,K],1,1);
        
            
    
        
    end
end













