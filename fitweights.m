function W=fitweights(Y,X,time,cvid,p, boot)
%options (p.regressionmethod): 'OLS', 'lasso', 'ridge'. standards. Also, 'FLS': flexible least squares where coefficients are encouraged to change slowly over time. Smoothness is controlled by p.flsmu
switch p.regressionmethod
    case 'FLS'
        persistent wfls
        %for FLS, we have to compute the weights considering all time
        %points, so we run this before looping over time and store the
        %result in a persistent variable.
        %Y = TxDxN
        %X = NxK
        %W = DxK(xT)(xB)
        if numel(size(Y))>2
            [T,D,N] = size(Y);
            K = size(X,2);
            wfls = zeros(D,K,T,p.nboot);

            for b = 1:p.nboot
                idx = boot.idx(:,b);
                Xb = squeeze(X(idx,:,boot.sets(b)));
                Yb = Y(:,:,idx);
                Xb=Xb';
                mu = p.flsmu;
                xxt = Xb*Xb'; %want KxK;

                for d = 1:D %god help you if you want to vectorize this
                    Q=zeros(K,K);
                    pe=zeros(K,1);
                    M=zeros(K,K,T-1);
                    em=zeros(K,1,T-1);
                    ys = squeeze(Yb(:,d,:))';%NxT
                    for t=1:T %forward sweep
                        %Q = Q_t-1, p = p_t-1
                        if t==T
                            wfls(d,:,T,b)=inv(Q+xxt)*(pe+Xb*ys(:,t));
                        else
                            M(:,:,t) = mu*inv(Q+mu*eye(K)+xxt);
                            em(:,:,t) = squeeze(M(:,:,t))*(pe+Xb*ys(:,t))*(1/mu);
                            Q=eye(K)-squeeze(M(:,:,t));
                            pe=mu*squeeze(em(:,:,t));
                        end
                    end
                    for t=fliplr(1:(T-1)) %backward sweep
                        wfls(d,:,t,b) = squeeze(em(:,:,t))+squeeze(M(:,:,t))*squeeze(wfls(d,:,t+1))';
                    end
                end

            end

        else
            W=squeeze(wfls(:,:,time,boot));
        end

    case 'lasso'
        D=size(Y,2);
        W = zeros(D, p.nchan);
        for i = 1:D
            [temp,s] = lasso(X, Y(:,i), 'CV', p.cvk);
            W(i,:) = temp(:, s.IndexMinMSE);
        end

    case 'ridge'
        D=size(Y,2);
        %need to do manual cv here I guess
        ridgelambdas = linspace(0,1 ,100);
        W = zeros(D, p.nchan);
        optimlambda = zeros(D,1);
        if isempty(cvid)
            N = size(Y,1);
            cvid = repmat([1:p.cvk], [1, floor(N/p.cvk)]);
        end
        for i = 1:D % I guess I'm fitting sepearate lambda for each electrode. I wonder if that's good ...
            temp = arrayfun(@(x) ridge(Y(cvid~=x,i), X(cvid~=x,:), ridgelambdas), 1:p.cvk, 'UniformOutput', false);
            ridgesses = arrayfun(@(x) sum(((X(cvid==x,:)*temp{x})-Y(cvid==x,i)).^2), 1:p.cvk, 'UniformOutput', false);
            sses = cell2mat(ridgesses');
            mses = mean(sses,1);
            optimlambda(i) = ridgelambdas(find(mses==min(mses)));
            W(i,:)=ridge(Y(:,i), X, optimlambda(i));
        end

    case 'OLS'
        W = (X\Y)';
end
end

% % %%wfls time courses
% figure
% for i = 1:8, plot(squeeze(wfls(50,i,:,1))); hold on, end