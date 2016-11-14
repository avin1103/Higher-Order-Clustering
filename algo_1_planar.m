%% Load Image
%img = double(imread('kinect_depth_image.jpg'));
img = double(imread('wall_person.jpg'));
img = imresize(img,[100,150]);
rows = size(img,1);
cols=size(img,2);
data = reshape(img,size(img,1)*size(img,2),3);
%imshow(img)


%%
N = rows*cols;
n = 5; % Number of points to build model
K = 4; % Define K

cluster = kmeans(data,K);
figure
pixel_labels = reshape(cluster,rows,cols);
imshow(pixel_labels,[]), title('image labeled by k-means');


%% Here comes the part of cluster initialisation
% Build P matrix.
num_column = 10; % Here we can see that number of columns are very small in comparision ti Nc
P=[];
pj=zeros(N,1);
for it=1:num_column
    r =randi([1,N],1,n);
    points = data(r,:); % n points from data which will make my model
    [nor,vec,poi] = affine_fit(points); 
    for iter = 1:N % Calculating Pj vector here
        temp=sum(r==iter);
        if temp==0
            X = data(iter,:);
            temp = X*nor;
            temp2 = poi*nor;
            err = (temp-temp2);
            sigma = 100;
            lembda = 10;
            pow=2;
            pj(iter) = exp(-lembda*(err/sigma)^pow);
        end
    end
    P = [P,pj];
end

cluster = kmeans(P,K);

figure 
pixel_labels = reshape(cluster,rows,cols);
imshow(pixel_labels,[]), title('image labeled by higher order k-means');

%% While loop 

T=1000; % number of column of P matrix (1<= T <= Nc)
numc = 4; % Number of column in U matrix
U = orth(randn(N,numc)); % Initialisation of U matrix
%err_1=100;
err_U=N*numc;
niter=3;
anku=1;
while ( err_U>10)% && err_1<N-5)
    prev_U = U;
    for t1 = 1:T
        
        r =randi([1,K],1,1);
        i = cluster==r;
        I = find(i==1);
        I=randsample(I,n); % Choosing random n points from 
        %I=I(1:n);
        points = data(I,:); % n points from data which will make my model
        [nor,vec,poi] = affine_fit(points);
        pj=zeros(N,1);
        for iter = 1:N % Calculating Pj vector here
            temp=sum(I==iter);
            if temp==0
                X = data(iter,:);
                temp = X*nor;
                temp2 = poi*nor;
                err = (temp-temp2); 
                sigma = 100;
                lembda = 10;
                pow=2;
                pj(iter) = exp(-lembda*(err/sigma)^pow);
            end
        end
        
        weights = U\pj;
        %weights = lsqnonneg(U,pj);
        residual = pj-U*weights;
        q = U*weights;
        norm_weights = norm(weights);
        norm_residual = norm(residual);
        norm_q = norm(q);
        sG = norm_residual*norm_q;
        step_size=0.001/anku;
        
        t = step_size*sG;
        %if t<pi/2 % drop big steps  
            alpha = (cos(t)-1)/(norm_q*norm_weights) ;
            beta = sin(t)/(norm_residual*norm_weights);
            U = U + beta*residual*weights' + alpha*q*weights';
        %end 
    end
    TF = isnan(U);
    if(sum(sum(TF))==N*numc)
        break
    else
        cluster = kmeans(U,K,'MaxIter',10000);
    end
   
    err_U = sum(sum(abs(prev_U-U)))
    niter=niter-1;
    anku=anku*5;
end
%%

figure
pixel_labels = reshape(cluster,rows,cols);
imshow(pixel_labels,[]), title('Image Labeled by My program...');
%clear;
%clc;

