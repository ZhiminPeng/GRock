function [b,y] = construct_bpdn_rhs(A,x,lambda,varargin)
%function [b,y] = construct_bdpn_rhs(A,x,lambda);
% For a given k times n matrix A, a give n-vector x and a given positive
% lambda, this function gives a k-vector b such that x is a solution of
%
%  min ||Ax - b||_2^2/2 + lambda ||x||_1.
%
% For the output y see below.
%
% The instance (A,b,x,lambda) can also be used as test instances for other
% reformulations of the Basis Pursuit Denoising. Namely, be setting tau =
% norm(x,1), the vector x is a solution of
%
%  min ||Ax - b||_2 s.t. ||x||_1 <= tau
%
% and for sigma = norm(A*x-b,2) the vector x is a solution of
%
%  min ||x||_1  s.t. ||Ax - b||_2 <= sigma.
%
% Input:
%  A: The given matrix.
%  x: The given "solution".
%  lambda: positive parameter.
%
% Optional input:
%  method: Chooses the method for solving the constraint minimization
%     problem. Can be either 'projected' (for the projected gradient method,
%     default), or 'conditional' (for the conditional gradient method).
%  maxIter: Maximal number of iteration for solving the constraint
%     minimization problem (default: 10*n).
%  tolres: Tolerance of the residual in the algorithm for the solution of
%     the constraint minimization problem (default: 1e-6).
%  tols: Another tolerance which is only needed for conditional gradient
%     method (somehow tolerance on the stepsize, default 1e-6).
%  Verbose: : no output on screen (0, default), lots of output (1).
%
% Output:
%   Two k-vectors b and y. b is as specified above and
%   y is such that A'*y is an element of the multivalued sign of x.
%   With the help of y one can contruct other valid right hands for
%   different values of lambda and difference vectors x which have the same
%   sign pattern by simply setting.
%     b = lambda*y + A*x.
%
% Example:
%  n=200; k=100; s = 20; lambda = 1;
%  A = randn(k,n);
%  x = zeros(n,1);
%  p = randperm(n);
%  x(p(1:s)) = randn(s,1);
%  [b,y] = construct_bpdn_rhs(A,x,lambda,'Verbose',1);
%  %Now check for optimality, e.g. by evalutating the fixed-point equation:
%  S = @(x,lambda) max(abs(x)-lambda,0).*sign(x);
%  norm(x - S(x-A'*(A*x-b),lambda))
%  %Construct a differeny b with another lambda:
%  lambda = 0.1;
%  b = lambda*y + A*x;
%  norm(x - S(x-A'*(A*x-b),lambda))
%
% @author: Dirk Lorenz, 08.02.2011, d.lorenz@tu-braunschweig.de
% @version: $Id: construct_bpdn_rhs.m 5 2011-02-09 09:11:51Z dirloren $

maxIter = [];
tolres = 1e-12;
tols = 1e-12;
method = 'pocs';
verbose = 0;
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch varargin{i}
     case 'maxIter'
       maxIter = varargin{i+1};
     case 'Verbose'
       verbose = varargin{i+1};
     case 'tolres'
       tolres = varargin{i+1};
     case 'tols'
       tols = varargin{i+1};
     case 'method'
       method = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
U = orth(A'); % Orthonormal basis for rg A'
P = U*U';     % Projection on rg A'


[~,n] = size(A);
s = length(find(x~=0));

if isempty(maxIter)
    maxIter = 10*n;
end

switch lower(method)
    case {'conditional','projected'}
        % Now construct w:
        w = zeros(n,1);
        w(x>0) = +lambda;
        w(x<0) = - lambda;
        % and v
        v = w(x~=0);

        En = eye(n);
        Pz = En(:,x==0);
        Pv = En(:,x~=0);


        Pbar = (P*Pz - Pz);
        vbar = (Pv - P*Pv)*v;

        % Now calculate z by solving P3*z = vv s.t. |z|<= lambda
        % with the conditional gradient method:
        z = zeros(n-s,1);

        if strcmp(method,'conditional')
            for iter =1:maxIter
                Pbarz = Pbar*z;
                res = Pbarz-vbar;
                p = -lambda*sign(Pbar'*(res));
                objval = norm(res);
                if verbose
                    fprintf('% 6d | Objective in cond grad: %e\n', iter, objval)
                end
                Pbarzp = Pbarz - Pbar*p;
                ss = (res)'*(Pbarzp)/norm(Pbarzp)^2;
                if objval < tolres*lambda && ss < tols*lambda
                    if verbose
                        fprintf('Conditional gradient method converged after %d iterations...\n',iter) 
                    end
                    break
                end
                z = z + min(ss,1)*(p-z);
            end
        elseif strcmp(method,'projected') %Or the projected gradient method:
            ss = normest(Pbar)^2;
            for iter = 1:maxIter
                res = Pbar*z-vbar;
                z = C(z-ss*Pbar'*(res),ss*lambda);
                objval = norm(res);
                if verbose
                    fprintf(' % 6d | Objective in proj grad: %e\n', iter, objval)
                end
                if objval < tolres*lambda
                    if verbose
                        fprintf('Projected gradient method converged after %d iterations...\n',iter) 
                    end
                    break
                end
            end
        end
        if objval>tolres*lambda
            error('No right hand side b found for this x. Try using a more sparse x.')
        end

        % Build w:
        w(x==0) = z;

        if verbose
            %Check, if w is in the range of A^T:
            fprintf('Is calculated w in rg A^T?: norm(Pw-w) = %e\n',norm(P*w-w))
            if norm(P*w-w)<tolres
                fprintf('   Yes!\n\n')
            else
                error('   No! Something went wrong...\n\n')
            end
        end

    case {'pocs'} % Projection onto convex sets
        w0 = zeros(n,1);
        w0(x>0) = +lambda;
        w0(x<0) = - lambda;
        w = w0;
        for iter=1:maxIter
            wold = w;
            Pwold = P*wold;
            error_rangeAt = norm(Pwold-wold);
            w = Ps(Pwold,w0,lambda);
            error_subgrad = norm(w - Pwold);
            if verbose
                fprintf('% 6d | pocs: distance to range: %4.2e, distance to subgrad: %4.2e\n',iter,error_rangeAt,error_subgrad)
            end
            if max(error_rangeAt,error_subgrad) < tolres
                if verbose
                    fprintf('pocs converged after %d iterations...\n',iter)
                end
                break
            end
        end
        if max(error_rangeAt,error_subgrad) >= tolres
            error('No right hand side b found for this x. Try using a more sparse x.')
        end
    otherwise
        error('Unknown method!')
end


% Calculate b:
y = A'\w;
if verbose
    fprintf('A^T*y = w solved with residuum %e \n', norm(A'*y-w))
end
b = y + A*x;
if verbose
    fprintf('b successfully constructed.\n\n')
end

function y = C(x,lambda)
% Cut-off function
y = sign(x).*min(abs(x),lambda);

function y = Ps(x,pattern,lambda)
% 
y = 0*x;
y(pattern > 0) = lambda;
y(pattern < 0) = -lambda;
y(pattern== 0) = C(x(pattern==0),lambda);