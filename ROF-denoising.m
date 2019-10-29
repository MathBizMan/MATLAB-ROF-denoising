	% code to perform ROF Algorithm in MATLAB
	
	lena512=imread('lena512.bmp');
	cam256=imread('cameraman256.bmp');
	%I=cam256;
	I=lena215;
	
	% add Gaussian noise, mean=0, variance=20^2
	I_noised=imnoise(I, 'gaussian',0,20^2/255^2);

	I=double(I);I_noised=double(I_noised);

	%% FPPO-ATV-B
	% generate matrix D and B, use the following two commands
	I_N=eye(N); % N is the size of images (N*N)
	I_N=sparse(I_N);%sparse representation
	A=kron(I_N, D);
	C=kron(D,I_N);
	B=[A; C]; Kronecker product of I_N and D

	%vectorization of the matrix
	x=I_noised(:);
	%%
	%%
	
	% reshape a vector into a matrix
	%I_denoise=-reshape(prox_x,[N,N]);

	% compute PSNR
	v=zeros(2*N^2,1); %initialization
	Iter=15; kappa=0; mu=50;
	BBt=B*B';
	for n = 1:Iter
		v_pre=v;
		y=B*x+v-lambda*BBt*v;
		% compute the proximity of 1/lambda*phi
		b=zeros(2*N^2,1);
		ind=find(abs(y)-mu/lambda>0);
		b(ind)=(abs(y(ind))-mu/lambda).*sign(y(ind));
		v=kappa*v+(1-kappa)*(y-b)
	end
	re=norm(v-v_pre,2)/norm(v,2)
	% compute prox_(phiB)x
	prox_x=x-lambda*B’*v;
	I_denoise=reshape(prox_x,[N,N]);
	figure(3)
	imshow(I_denoise,[]);
	% compute PSNR
	MSE=sum(sum((I-I_denoise).^2))/(N*N);
	PSNR=20*log10(255/sqrt(MSE));
	subplot(1,3,1),imshow(I,[]);subplot(1,3,2),imshow(I_noised,[]);subplot(1,3,3),imshow(I_denoise,[])
	PSNR
