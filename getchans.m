function chanids = getchans(chanlocs, in)

%cap montages can be found here https://www.brainproducts.com/downloads/cap-montages/
%input {[i d s of rows], 'optional: left or right'};
%9 rows, indexed posterior to anterior

    chanids = 1:numel(chanlocs);
    if isempty(in)
        return
    end
    if isnumeric(in)
        chanids = chanids(in);
        return
    end
    if ~iscell(in)
        in = eval(in);
    end
    chansbyrow = {{'O1', 'Oz', 'O2'}, ... %back row left to right
                {'PO7', 'PO3', 'POz', 'PO4', 'PO8'}, ... %second row left to right
                {'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'}, ... %etc.
                {'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10'}, ...
                {'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8'}, ...
                {'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10'}, ...
                {'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'}, ...
                {'AF7', 'AF3', 'AFz', 'AF4', 'AF8'}, ...
                {'Fp1', 'Fp2'}};
    
    nums = cellfun(@(x) isnumeric(x), in);
    
    chanlist = cat(1,[chansbyrow{in{nums}}]);

    leftright = cellfun(@(x) ~isnumeric(x), in);
    if any(leftright)
        switch in{leftright}
            case 'left'
                subset = cellfun(@(x) ~isempty(x), regexp(chanlist, '[13579]$'));
                chanlist = chanlist{subset};
            case 'right'
                subset = cellfun(@(x) ~isempty(x), regexp(chanlist, '[24680]$'));
                chanlist = chanlist{subset};
        end
    end

    chanids = find(ismember({chanlocs(:).labels}, chanlist));

end