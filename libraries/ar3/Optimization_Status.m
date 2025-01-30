classdef Optimization_Status < int32
    enumeration
        RUNNING (1)
        SUCCESS (0)
        MAX_ITERATIONS_EXCEEDED (-1)
        NUMERICAL_ISSUES (-2)
        ILL_CONDITIONED (-3)
        NOT_LOWER_BOUNDED (-4)
        USER_TERMINATED (-100)
    end
end
