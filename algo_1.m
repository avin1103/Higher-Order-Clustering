%% Load Input
%Creating Data
num_circle=10; %number of circles 
num_points = 100; %number of points for in given circle
s=5;% number of segnments that we will initially cluster
r=50; %radius is fixed for all circles, my be changed
angles = 2*pi*rand(1,num_points); %anglular poistion of a point in a circle
%Dimension=2; % NUmber of dimension
data = [];
figure
for i=1:num_circle
  cx= rand*100;  %a random x center coordinate
  cy= rand*100;  %a random y center coordinate
  x=r*cos(angles)+cx;  %parametric equation of a circle
  y=r*sin(angles)+cy;
  c=(randi([1, s]))/10; 
  color=c.*[0.25,0.5,0];
  temp = [x;y];
  data = [data, temp];
  scatter(x,y,'MarkerFaceColor',color); title 'Initial Dataset';
  hold on;
end
data=data';
%clearvars x y temp cx cy color c angles r s i;
%%
N = num_circle*num_points;
n = 4; % Number of points to build model

K = num_circle;

cluster = kmeans(data,K);
figure 
for jhg=1:K
    i = cluster==jhg;
    I = find(i==1);
    points = data(I,:)';
    scatter(points(1,:),points(2,:),'MarkerFaceColor',color); title 'Kmeans Classification';
    hold on;
end

%% Here comes the part of cluster initialisation
% Build P matrix.
num_column = 100;
P=[];
pj=zeros(N,1);
for it=1:num_column
    r =randi([1,N],1,n);
    points = data(r,:); % n points from data which will make my model
    points = points';
    %[cx,cy,r,error] = circFit(points);
    [cx, cy, rad, error] = CIRC(points(1,:)',points(2,:)');
    for iter = 1:N % Calculating Pj vector here
        temp=sum(r==iter);
        if temp==0
            x = data(iter,1);
            y = data(iter,2);
            X = [x,y;cx,cy];
            d = pdist(X,'euclidean');
            err = sqrt(abs(d^2-rad^2)); 
            sigma = 10;
            lembda = 1;
            pow=2;
            pj(iter) = exp(-lembda*(err/sigma)^pow);
        end
    end
    P = [P,pj];
end
clear err;
cluster = kmeans(P,K);
figure 
for jhg=1:K
    i = cluster==jhg;
    I = find(i==1);
    points = data(I,:)';
    scatter(points(1,:),points(2,:),'MarkerFaceColor',color); title 'Higher Order Kmeans Classification';
    hold on;
end

%% While loop 

T=100; % number of column of P matrix (1<= T <= Nc)
numc = 5; % Number of column in U matrix
niter=10;
U = orth(randn(N,numc)); % Initialisation of U matrix
err_1=100;
err_U=N*numc;
while (err_U>0.01*N*numc && err_1>5 )% && err_1<N-5)
    prev_U = U;
    for t1 = 1:T
        r =randi([1,K],1,1);
        i = cluster==r;
        I = find(i==1);
        I=randsample(I,n); % Choosing random n points from 
        %I=I(1:n);
        points = data(I,:); % n points from data which will make my model
        points = points';
        %[cx,cy,r,error] = circFit(points); % This is the circular fit.
        [cx, cy, rad, error] = CIRC(points(1,:)',points(2,:)');
        pj=zeros(N,1);
        for iter = 1:N % Calculating Pj vector here
         %iter=1;
        % while(iter<=N)
            temp=sum(I==iter);
            if temp==0
                x = data(iter,1);
                y = data(iter,2);
                X = [x,y;cx,cy];
                d = pdist(X,'euclidean');
                err = sqrt(abs(d^2-rad^2)); 
                sigma = 10;
                lembda = 10;
                pow=2;
                pj(iter) = exp(-lembda*(err/sigma)^pow);
            end
            %iter=iter+5;
        end
        
        weights = U\pj;
        %weights = lsqnonneg(U,pj);
        residual = pj-U*weights;
        q = U*weights;
        norm_q = norm(q);
        norm_weights = norm(weights);
        norm_residual = norm(residual);
        
        sG = norm_residual*norm_q;
        step_size=1/(t1+1);
        t = step_size*sG;
        if t<pi/2 % drop big steps  
            alpha = (cos(t)-1)/(norm_q*norm_weights) ;
            beta = sin(t)/(norm_residual*norm_weights);
            U = U + beta*residual*weights' + alpha*q*weights';
        end 
    end
    prev_cluster=cluster;
    TF = isnan(U);
    if(sum(sum(TF))==N*numc)
        break
    else
        cluster = kmeans(U,K);
    end
    prev_err_1 = err_1;
    err_1 = sum(abs(prev_cluster-cluster));
   % if(abs(prev_err_1-err_1)<5*num_circle)
        %err_1=abs(prev_err_1-err_1);
     %   err_1=4;
    %end
    err_U = sum(sum(abs(prev_U-U)))
    niter = niter-1;
    
    clearvars x y X d sigma err i I cx cy r prev_cluster;
end


figure
    for jhg=1:K
        i = cluster==jhg;
        I = find(i==1);
        points = data(I,:)';
        scatter(points(1,:),points(2,:),'MarkerFaceColor',color);title 'My Program Classification';
        hold on
    end
clearvars i I jhg norm_q norm_residual norm_weights norm_circle;
clear ;
%clc;
