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
cluster = randi([1 K],1,N);
cluster=cluster'; % This is my uniform random sampling
%% While loop 
n = 5; % Number of points to build model
error = 1000;
T =100; % number of column of P matrix (1<= T <= Nc)
while error>=1000
    for t = 1:1
        r =randi([1,K],1,1);
        i = cluster==r;
        I = find(i==1);
        I=I(1:n); % Choosing first n points from 
        % Make a model on these n points
        points = data(I,:); % n points from data which will make my model
        points = points';
        [cx,cy,r,error] = circFit(points); % This is the circulur fit.
        pj=zeros(N,1);
        for iter = 1:N % Calculating Pj vector here
            temp=sum(I==iter)
            
            if temp==0
                x = data(iter,1);
                y = data(iter,2);
                X = [x,y;cx,cy];
                d = pdist(X,'euclidean');
                err = abs(d-r); 
                sigma = 10;
                pj(iter) = exp(-err/sigma);
            end
        end
        
        clearvars x y X d sigma err;
        
        
        
        
        
        error=error-1;
    
    end
end





