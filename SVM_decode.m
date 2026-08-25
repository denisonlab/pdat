function [outp, preds] = SVM_decode(t, train, test, p, ttest,lambda_var, lambda)
%outp: final prediction
%preds: predictions within each bagging iteration
if nargin<5
    ttest = t;
end

train_samples = squeeze(train.allsamples(t,:,:));
test_samples = squeeze(test.allsamples(ttest,:,:));
% if p.loocv
%     %squeeze(1x64x1) -> 1x64
%     %squeeze(1x64xN) -> 64xN
%     if size(test_samples,2)~=1
%         test_samples = test_samples';
%     end
%     Ntesttrials = 1;
% else
%     Ntesttrials = size(test_samples,1);
% end
D = size(train_samples,1);
N = size(train_samples, 2);

testdim = find(size(test_samples)==D);
if testdim~=1
    test_samples = test_samples';
end
Ntesttrials = size(test_samples,2);

preds = zeros(p.nboot, Ntesttrials);

if p.svmmethod == "class"
    [classes, classids,train.class] = unique(train.stimval);
    [~, ~,test.class] = unique(test.stimval);

    for b=1:p.nboot
        idx = train.boot.idx(:,b);
        boot_samples = train_samples(:,idx);

        model = fitcecoc(boot_samples', train.class(idx), 'Coding', 'onevsall');
        predicted_label = predict(model, test_samples);
        preds(b,:) = classes(predicted_label);

    end
    if p.iscircular
        outp = circ_mean(preds, [],1);
    else
        outp = mean(preds, 1);
    end
elseif p.svmmethod == "SVR"
    if p.iscircular
        for b = 1:p.nboot
            idx = train.boot.idx(:,b);
            boot_samples = train_samples(:,idx);
            boot_stimval = train.stimval(idx);
            sinmod = fitrlinear([boot_samples; ones(1,N)]', sin(boot_stimval));
            cosmod = fitrlinear([boot_samples; ones(1,N)]', cos(boot_stimval));

            try
                sinpred = predict(sinmod, [test_samples; ones(1, Ntesttrials)]');
                cospred = predict(cosmod, [test_samples; ones(1, Ntesttrials)]');
            catch
                disp(size(test_samples));
                disp(Ntesttrials);
                sinpred = predict(sinmod, [test_samples'; ones(1, Ntesttrials)]');
                cospred = predict(cosmod, [test_samples'; ones(1, Ntesttrials)]');
            end

            preds(b,:) = atan2(sinpred, cospred);
            %distance to the hyperplane?
            %doesn't make sense in terms of the circular variable, but you
            %could do sqrt(sin^2+cos^2)

        end
        outp = circ_mean(preds, [],1);
    else
        for b = 1:p.nboot
            idx = train.boot.idx(:,b);
            boot_samples = train_samples(:,idx);
            boot_stimval = train.stimval(idx);
            mod = fitrlinear([boot_samples; ones(1,N)]', boot_stimval);
            prediction = predict(mod, [test_samples; ones(1, Ntesttrials)]');

            preds(b,:) = prediction;
        end
        outp = mean(preds, 1);
    end

end

end

