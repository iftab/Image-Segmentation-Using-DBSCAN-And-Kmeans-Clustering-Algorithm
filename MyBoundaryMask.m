function BW = MyBoundaryMask(L,varargin)
narginchk(1,2);

numericTypes = images.internal.iptnumerictypes();

validateattributes(L,{numericTypes{:},'logical'},...
    {'finite','2d','nonnegative','nonsparse'},mfilename); %#ok<CCAT>

if nargin < 2
    conn = 8; % Default connectivity is 8.
else
    connIn = varargin{1};
    validateattributes(connIn,numericTypes,{'scalar','finite','positive'},...
        mfilename,'CONN');
    coder.internal.errorIf(~coder.internal.isConst(connIn), ...
        'MATLAB:images:validate:codegenInputNotConst','CONN');
    conn = double(connIn);
end

coder.internal.errorIf(~isequal(conn,4) && ~isequal(conn,8), ...
    'images:boundarymask:invalidConnectivity');

if conn == 4
    se = [0,1,0; 1,1,1; 0,1,0];
else
    se=strel('diamond',1);
end

BW = (imdilate(L,se) > L) | (imerode(L,se) < L);
