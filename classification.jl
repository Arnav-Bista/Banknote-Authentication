using Plots
using CSV
using Tables
using Random

function load(path = "data_banknote_authentication.csv", shuf = false)
    a = CSV.File(path) |> Tables.matrix
    if shuf
        a = a[shuffle(1:end),:]
    end
    m, p = size(a)
    X = fill(1.0,m,p)
    X[:,2:p] = a[:,1:4]
    Y = a[:,5]
    return X,Y
end

function sigmoid(Z)
    return 1/(1 + exp(-Z))
end

function LogLoss(theta)
    global X,Y
    m = size(Y)[1]
    h = sigmoid.(X*theta)
    return  (-1 / m) * sum((Y .* log10.(h)) .+ ((1 .- Y) .* log10.(1 .- h)))
end

function BGD(x,y,theta)
    m = size(y)[1]
    a = 0.001
    iterations = 10000
    J = [0.0 for i in 1:iterations]
    for i in 1:iterations
        h = sigmoid.(x*theta)
        theta = theta + a/m * transpose(x) * (y .- h)
        J[i] = LogLoss(theta)
    end
    return theta,J
end

function Test(X,Y,theta)
    h = sigmoid.(X*theta)
    m = size(Y)[1]
    c = 0
    for i in 1:m
        if h[i] > 0.5
            h[i] = 1
        else
            h[i] = 0
        end
        if h[i] == Y[i]
            c = c + 1
        end
    end
    return round(c/m * 100,digits = 3)
end

X,Y = load("data_banknote_authentication.csv")
t = [0 for i in 1:size(X)[2]]
println("Initial Costs: ", LogLoss(t))
theta,J =BGD(X,Y,t)
# a = @elapsed BGD(X,Y,t)
println(Test(X,Y,theta))
