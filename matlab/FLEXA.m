function [x, hist] = FLEXA(b,lambda, A, xs, rho, varargin)

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch varargin{i}
            case 'maxIter'
                maxIter = varargin{i+1};
            case 'Verbose'
                verbose = varargin{i+1};
            case 'Positivity'
                positivity = varargin{i+1};
            case 'Tol_error'
                tol_error = varargin{i+1};
            case 'Tol_obj'
                tol_obj = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end                                            

%% initialize parameters
t_start = tic;
[m,n] = size(A);
hist.obj = [];
hist.rel_err = [];
hist.rel_obj = [];
hist.time    = [];
hist.updateNumber = [];
norm_xs = norm(xs);
opt_obj = 0.5* norm(A*xs - b)^2 + lambda * norm(xs, 1);

%% compute the two norm of the columns of A
normA = zeros(n,1);
for i = 1:n
    normA(i) = norm(A(:,i));
end

x0 = zeros(n,1);
x = x0;
%tau = 0.0001;
tau = trace(A'*A)/(2*n);
gamma = 0.9;
theta = 1e-7;
%% main loop
for iter = 1:maxIter
    x0 = x;
    %% calculate the potential update
    res = b - A*x0;
    beta = A'*res + normA.*normA.*x0 + tau * x0;
    xAll = sign(beta)./(normA.^2 + tau).* max(abs(beta) - lambda,0);
    %% select some indexes to update
    d = xAll - x0;
    M = max(abs(d));
    updateIdxSet = find(abs(d)>=rho*M);
    z =x0;
    z(updateIdxSet) = xAll(updateIdxSet);
    %% choose a stepsize to update
    obj = 0.5 * norm(A*x0 - b)^2 + lambda*norm(x0, 1);
    rel_obj = (obj - opt_obj)/opt_obj;
    gamma = gamma*(1 - min(1, 1e-4/rel_obj)*theta * gamma);
    x = x0 + gamma * (z - x0);
    %% dynamically update tau
    obj_new = 0.5 * norm(A*x - b)^2 + lambda*norm(x, 1);
    if(iter < 100)
        if(obj_new > obj)
            tau = tau * 2;
            x = x0;
        elseif(rel_obj <=0.01 || (length(hist.rel_obj)>10 && ...
                sum(hist.rel_obj(end-9:end)-hist.rel_obj(end-10:end-1)<0)==10))
            tau = tau/2;
            rel_err = norm(xs-x)/norm_xs;
        end
    else
        rel_err = norm(xs-x)/norm_xs;
    end
    if(mod(iter, 10)==0)
        disp(sprintf('tau=%e \t gamma=%e \t\n', tau, gamma));
    end
    rel_err = norm(xs-x)/norm_xs;
    t_end = toc(t_start);
        %% record the history of the iteration
    hist.rel_obj = [hist.rel_obj;rel_obj];
    hist.rel_err = [hist.rel_err; rel_err];
    hist.updateNumber = [hist.updateNumber; length(updateIdxSet)];
    hist.time = [hist.time; t_end];
    %% stopping criterion check
    if(rel_obj<tol_obj && rel_err < tol_error)
        break;
    end
    
end

end