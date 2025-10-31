function [s, varargout] = mgh_wrapper(varargin)
    throw(MException('MGH:NotCompiled', 'The MGH wrapper needs to be compiled first.'));
end
