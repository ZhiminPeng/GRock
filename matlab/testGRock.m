%% Test GRock

clc; clear all;
addpath ../Data/;
%% generate data
lambda = 1;
m = 2048;
n = 4096;
k = n * 0.1;
options.sparsity = k;
options.dim = [m, n];
foo = ['dynamic1';'dynamic2';'dynamic3';'dynamic4';'dynamic5';'dynamic6';'dynamic7'];
options.seed = 1;
j = 1;
[A,b,xdagger,tau,sigma,~] = construct_bpdn_instance('gauss',...
    foo(j,:), lambda, options);
optobj = 0.5*sigma^2 + lambda*tau;

maxIter = 100;
tol_error = 1e-6;
tol_obj = 1e-10;

%% Test with FLEXA with rho = 0.5
disp('start FLEXA with rho = 0.5')
rho = 0.5;
[flex_u1, flex_hist1] = ...
    FLEXA(b,lambda,A, xdagger, rho,...
    'maxIter',maxIter,...
    'Verbose',0,...
    'Tol_error',tol_error,...
    'Tol_obj',tol_obj);

%% Test with FLEXA with rho = 0
disp('start FLEXA with rho = 0')
rho = 0;
[flex_u2, flex_hist2] = ...
    FLEXA(b,lambda,A, xdagger, rho,...
    'maxIter',maxIter,...
    'Verbose',0,...
    'Tol_error',tol_error,...
    'Tol_obj',tol_obj);

%% GRock with dynamic update of P
opts.maxIter = maxIter;
opts.xs = xdagger;
opts.obj_tol = tol_obj;
opts.rel_tol = tol_error;
[x1, hist_GRock_d] = GRock_LASSO(A, b, lambda, opts);

%% GRock with fixed P
opts.maxIter = maxIter;
opts.P = 20;
opts.mode = 'fixed';
[x2, hist_GRock_f] = GRock_LASSO(A, b, lambda, opts);


%% plot the curves
figure(1);
subplot(3,1,1);
semilogy(hist_GRock_d.time, hist_GRock_d.rel_err,'r',...
    hist_GRock_f.time, hist_GRock_f.rel_err,'r--',...
    flex_hist1.time, flex_hist1.rel_err,'k',...
    flex_hist2.time, flex_hist2.rel_err,'k--',...
    'linewidth', 2);
title('relative error');
legend('GRock with dynamic P','GRock with fixed P', 'FLEXA sigma = 0.5','FLEXA sigma = 0');
subplot(3,1,2);
semilogy(hist_GRock_d.time, hist_GRock_d.obj_err,'r',...
    hist_GRock_f.time, hist_GRock_f.obj_err,'r--',...
     flex_hist1.time, flex_hist1.rel_obj,'k',...
     flex_hist2.time, flex_hist2.rel_obj,'k--',...
    'linewidth', 2);
title('function error');
legend('GRock with dynamic P','GRock with fixed P', 'FLEXA sigma = 0.5','FLEXA sigma = 0');

subplot(3,1,3);
plot(1:length(hist_GRock_d.P), hist_GRock_d.P,'r',...
    1:length(flex_hist1.updateNumber),flex_hist1.updateNumber,'k',...
'linewidth',2);
title('# of updates in each iteration');
legend('Dynamic update P', 'FLEXA sigma = 0.5');


figure(2);
subplot(3,1,1);
semilogy(1:length(hist_GRock_d.rel_err), hist_GRock_d.rel_err,'r',...
    1:length(hist_GRock_f.rel_err), hist_GRock_f.rel_err,'r--',...
    1:length(flex_hist1.rel_err), flex_hist1.rel_err,'k',...
    1:length(flex_hist2.rel_err), flex_hist2.rel_err,'k--',...
    'linewidth', 2);
title('relative error');
legend('GRock with dynamic P','GRock with fixed P', 'FLEXA sigma = 0.5','FLEXA sigma = 0');
subplot(3,1,2);
semilogy(1:length(hist_GRock_d.rel_err), hist_GRock_d.obj_err,'r',...
    1:length(hist_GRock_f.rel_err), hist_GRock_f.obj_err,'r--',...
     1:length(flex_hist1.rel_err), flex_hist1.rel_obj,'k',...
     1:length(flex_hist2.rel_err), flex_hist2.rel_obj,'k--',...
    'linewidth', 2);
title('function error');
legend('GRock with dynamic P','GRock with fixed P', 'FLEXA sigma = 0.5','FLEXA sigma = 0');

subplot(3,1,3);
plot(1:length(hist_GRock_d.P), hist_GRock_d.P,'r',...
    1:length(flex_hist1.updateNumber),flex_hist1.updateNumber,'k',...
'linewidth',2);
title('# of updates in each iteration');
legend('Dynamic update P', 'FLEXA sigma = 0.5');

