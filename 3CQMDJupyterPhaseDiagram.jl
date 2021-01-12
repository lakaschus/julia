@time using BenchmarkTools
@time using DifferentialEquations
@time using Distributed
@time using Dierckx
@time using DelimitedFiles
#using IJulia
#IJulia.installkernel("Julia 6 Threads", env=Dict(
#    "JULIA_NUM_THREADS" => "6",
#))
# https://phrb.github.io/2019-02-16-intro_parallel_julia/
# https://codingclubuc3m.github.io/2018-06-06-Parallel-computing-Julia.html
# https://discourse.julialang.org/t/setting-julia-num-threads-in-an-ijulia-kernel/8301

println(Threads.nthreads())

function sig_diq_mat_sum(k, ux, uxx, uy, uyy, uxy, x, y, T, mu)
  sqEpi = k .^ 2 .+ 2 .* ux 
  sqEs = sqEpi .+ 4 .* x .* uxx
  sqENG = k .^ 2 .+ 2 .* uy
  sqEG = sqENG .+ 4 .* y .* uyy
  alpha0 = 16 .* mu .^ 4 .- 4 .* mu .^ 2 .* sqEG .- 4 .* mu .^ 2 .* sqENG .+ sqEG .* sqENG .- 8 .* mu .^ 2 .* sqEs .+ sqEG .* sqEs .+ sqENG .* sqEs .- 16 .* uxy .^ 2 .* x .* y .+ 0im
  alpha1 = 2 .* sqEG .+ 2 .* sqENG .+ 2 .* sqEs .+ 0im
  alpha2 = 3 .+ 0im
  beta0 = (.-4 .* mu .^ 2 .+ sqENG) .* (.-4 .* mu .^ 2 .* sqEs .+ sqEG .* sqEs .- 16 .* uxy .^ 2 .* x .* y) .+ 0im
  beta1 = 16 .* mu .^ 4 .- 4 .* mu .^ 2 .* (sqEG .+ sqENG .- 2 .* sqEs) .+ sqENG .* sqEs .+ sqEG .* (sqENG .+ sqEs) .- 16 .* uxy .^ 2 .* x .* y .+ 0im
  beta2 = 8 .* mu .^ 2 .+ sqEG .+ sqENG .+ sqEs .+ 0im
  beta3 = 1 .+ 0im

  var1 = ( ( 27  .*  ( beta0 ) .^ ( 2 ) .+ ( 4  .*  ( beta1 ) .^ ( 3 ) .+ ( .-18  .*  beta0  .*  beta1  .*  beta2 .+ ( .-1  .*  ( beta1 ) .^ ( 2 )  .*  ( beta2 ) .^ ( 2 ) .+ 4  .*  beta0  .*  ( beta2 ) .^ ( 3 ) ) ) ) ) ) .^ ( 1/2 )
  var2 = ( .-27  .*  beta0 .+ ( 9  .*  beta1  .*  beta2 .+ .-2  .*  ( beta2 ) .^ ( 3 ) ) )

  z1 = ( ( 6 ) .^ ( .-1/2 )  .*  ( ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( .-1/3 )  .*  ( ( ( 6  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ 2  .*  var2 ) ) .^ ( 2/3 ) .+ ( .-6  .*  ( 2 ) .^ ( 1/3 )  .*  beta1 .+ ( .-2  .*  ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( 1/3 )  .*  beta2 .+ 2  .*  ( 2 ) .^ ( 1/3 )  .*  ( beta2 ) .^ ( 2 ) ) ) ) ) .^ ( 1/2 ) )
  z2 = (1/2  .*  ( 3 ) .^ ( .-1/2 )  .*  ( ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( .-1/3 )  .*  ( complex( 0,1 )  .*  ( complex( 0,1 ) .+ ( 3 ) .^ ( 1/2 ) )  .*  ( ( 6  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ 2  .*  var2 ) ) .^ ( 2/3 ) .+ ( 6  .*  ( 2 ) .^ ( 1/3 )  .*  ( 1 .+ complex( 0,1 )  .*  ( 3 ) .^ ( 1/2 ) )  .*  beta1 .+ ( .-4  .*  ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( 1/3 )  .*  beta2 .+ complex( 0,.-2 )  .*  ( 2 ) .^ ( 1/3 )  .*  ( complex( 0,.-1 ) .+ ( 3 ) .^ ( 1/2 ) )  .*  ( beta2 ) .^ ( 2 ) ) ) ) ) .^ ( 1/2 ))
  z3 = (1/2  .*  ( 3 ) .^ ( .-1/2 )  .*  ( ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( .-1/3 )  .*  ( ( .-1 .+ complex( 0,.-1 )  .*  ( 3 ) .^ ( 1/2 ) )  .*  ( ( 6  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ 2  .*  var2 ) ) .^ ( 2/3 ) .+ ( 6  .*  ( 2 ) .^ ( 1/3 )  .*  ( 1 .+ complex( 0,.-1 )  .*  ( 3 ) .^ ( 1/2 ) )  .*  beta1 .+ ( .-4  .*  ( ( 3  .*  ( 3 ) .^ ( 1/2 )  .*  var1 .+ var2 ) ) .^ ( 1/3 )  .*  beta2 .+ complex( 0,2 )  .*  ( 2 ) .^ ( 1/3 )  .*  ( complex( 0,1 ) .+ ( 3 ) .^ ( 1/2 ) )  .*  ( beta2 ) .^ ( 2 ) ) ) ) ) .^ ( 1/2 ))

  res = ( -1/2  .*  ( z1 ) .^ ( -1 )  .*  ( ( -1  .*  ( z1 ) .^ ( 2 ) .+ ( z2 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( ( -1  .*  ( z1 ) .^ ( 2 ) .+ ( z3 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( alpha0 .+ ( ( z1 ) .^ ( 2 )  .*  alpha1 .+ ( z1 ) .^ ( 4 )  .*  alpha2 ) )  .*  
	 tan.( 1/2  .*  ( T ) .^ ( -1 )  .*  z1 ).^(-1) .+ ( -1/2  .*  ( z2 ) .^ ( -1 )  .*  ( ( ( z1 ) .^ ( 2 ) .-  ( z2 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( ( -1  .*  ( z2 ) .^ ( 2 ) .+ ( z3 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( alpha0 .+ ( ( z2 ) .^ ( 2 )  .*  alpha1 .+ ( z2 ) .^ ( 4 )  .*  alpha2 ) )  .*  tan.( 1/2  .*  ( T ) .^ ( -1 )  .*  z2 ).^(-1) .+ (-1/2)  .*  ( z3 ) .^ ( -1 )  .*  ( ( ( z1 ) .^ ( 2 ) .+ (-1)  .*  ( z3 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( ( ( z2 ) .^ ( 2 ) .- 1  .*  ( z3 ) .^ ( 2 ) ) ) .^ ( -1 )  .*  ( alpha0 .+ ( ( z3 ) .^ ( 2 )  .*  alpha1 .+ ( z3 ) .^ ( 4 )  .*  alpha2 ) )  .*  tan.( 1/2  .*  ( T ) .^ ( -1 )  .*  z3 ).^(-1) ) )
  
  return real(res)
end

function Epion(ux, k)
  return sqrt.(k^2 .+ 2*ux)
end

function Esig(x, ux, uxx, k)
  return sqrt.(k^2 .+ 2*ux .+ 4*x.*uxx)
end

function Eq(x, k)
  return sqrt.(k^2 .+ hx^2*x) 
end

function Epsi(x, y, k, mu, n)
  return sqrt.(hy^2*y .+ ( Eq(x, k) .+ (-1)^n*mu ).^2)
end

function ENG(uy, k)
  return sqrt.(k^2 .+ 2*uy)
end

function fd(u, dx, dy, N, Nd)
    uxForw = (-3.0/2.0*u[1] + 2.0*u[2] - 1.0/2.0*u[3])/dx                                           
    uxBack = (3.0/2.0*u[N] - 2.0*u[N-1] + 1.0/2.0*u[N-2])/dx                                      
    uxCent = (-1.0/2.0*u[1:N-2] + 1.0/2.0*u[3:N])/dx                                                
    ux = append!(append!([uxForw], uxCent), [uxBack])                                               
    
    for i=N+1:N:N*Nd                                                                     
        uxForw = (-3.0/2.0*u[i] + 2.0*u[1+i] - 1.0/2.0*u[2+i])/dx                                   
        uxBack = (3.0/2.0*u[i+N-1] - 2.0*u[i+N-2] + 1.0/2.0*u[i+N-3])/dx                            
        uxCent = (-1.0/2.0*u[i:N+i-3] + 1.0/2.0*u[i+2:N+i-1])/dx                                      
	ux = append!(ux, append!(append!([uxForw], uxCent), [uxBack]))  
    end
    uxxForw = (-3.0/2.0*ux[1] + 2.0*ux[2] - 1.0/2.0*ux[3])/dx                                           
    uxxBack = (3.0/2.0*ux[N] - 2.0*ux[N-1] + 1.0/2.0*ux[N-2])/dx                                      
    uxxCent = (-1.0/2.0*ux[1:N-2] + 1.0/2.0*ux[3:N])/dx                                                
    uxx = append!(append!([uxxForw], uxxCent), [uxxBack])                                               
    
    for i=N+1:N:N*Nd                                                                     
        uxxForw = (-3.0/2.0*ux[i] + 2.0*ux[1+i] - 1.0/2.0*ux[2+i])/dx                                   
        uxxBack = (3.0/2.0*ux[i+N-1] - 2.0*ux[i+N-2] + 1.0/2.0*ux[i+N-3])/dx                            
        uxxCent = (-1.0/2.0*ux[i:N+i-3] + 1.0/2.0*ux[i+2:N+i-1])/dx                                      
	uxx = append!(uxx, append!(append!([uxxForw], uxxCent), [uxxBack]))  
    end

    uyForw = (-3.0/2.0*u[1:N] + 2.0*u[N+1:2*N] - 1.0/2.0*u[2*N+1:3*N])/dy
    uyBack = (3.0/2.0*u[(Nd-1)*N+1:Nd*N] - 2.0*u[(Nd-2)*N+1:(Nd-1)*N] + 1.0/2.0*u[(Nd-3)*N+1:(Nd-2)*N])/dy
    uyCent = (-1.0/2.0*u[1:(Nd-2)*N] + 1.0/2.0*u[2*N+1:Nd*N])/dy
    uy = append!(append!(uyForw, uyCent), uyBack)
    
    uyyForw = (-3.0/2.0*uy[1:N] + 2.0*uy[N+1:2*N] - 1.0/2.0*uy[2*N+1:3*N])/dy
    uyyBack = (3.0/2.0*uy[(Nd-1)*N+1:Nd*N] - 2.0*uy[(Nd-2)*N+1:(Nd-1)*N] + 1.0/2.0*uy[(Nd-3)*N+1:(Nd-2)*N])/dy
    uyyCent = (-1.0/2.0*uy[1:(Nd-2)*N] + 1.0/2.0*uy[2*N+1:Nd*N])/dy
    uyy = append!(append!(uyyForw, uyyCent), uyyBack)

    uxyForw = (-3.0/2.0*ux[1:N] + 2.0*ux[N+1:2*N] - 1.0/2.0*ux[2*N+1:3*N])/dy
    uxyBack = (3.0/2.0*ux[(Nd-1)*N+1:Nd*N] - 2.0*ux[(Nd-2)*N+1:(Nd-1)*N] + 1.0/2.0*ux[(Nd-3)*N+1:(Nd-2)*N])/dy
    uxyCent = (-1.0/2.0*ux[1:(Nd-2)*N] + 1.0/2.0*ux[2*N+1:Nd*N])/dy
    uxy = append!(append!(uxyForw, uxyCent), uxyBack)

    return ux, uxx, uy, uyy, uxy
end

function dudk(u, p, t)
  global counter, progress_steps
  counter += 1
  if counter == progress_steps
    println("flow scale k: ", t)
    counter = 1
  end
  k = t
  T, mu = p
  ux, uxx, uy, uyy, uxy = fd(u, dx, dy, Nx, Ny)
  Ep = Epion(ux, k)
  #Es = Esig(xx, ux, uxx, k)
  Eng = ENG(uy, k)
  Eps = Eq(xx, k)
  Ek1 = Epsi(xx, yy, k, mu, 1)
  Ek2 = Epsi(xx, yy, k, mu, 0)
  dudk = k^4/(12*pi^2)*(3.0*(Ep.^(-1)) .*(coth.(Ep/(2*T))) .+ 2*(Eng.^(-1)) .*(2*sinh.(Eng/T)) .*((cosh.(Eng/T) .- cosh.((2*mu)/T)).^(-1))
			.- 8*(Ek1.^(-1)) .*(1 .- mu*(Eps.^(-1))) .*tanh.(Ek1/(2*T)) .- 8*(Ek2.^(-1)) .*(1 .+ mu*(Eps.^(-1))) .*tanh.(Ek2/(2*T))
			.- 4*(1*Eps.^(-1) .*(tanh.((Eps .- mu)/(2*T)) .+ tanh.((Eps .+ mu)/(2*T))))
           .+ 2*sig_diq_mat_sum.(k, ux, uxx, uy, uyy, uxy, xx, yy, T, mu) )
  return dudk
end

function dudk_par(u, p, t)
  k = t
  T, mu = p
  ux, uxx, uy, uyy, uxy = fd(u, dx, dy, Nx, Ny)
  Ep = Epion(ux, k)
  #Es = Esig(xx, ux, uxx, k)
  Eng = ENG(uy, k)
  Eps = Eq(xx, k)
  Ek1 = Epsi(xx, yy, k, mu, 1)
  Ek2 = Epsi(xx, yy, k, mu, 0)
  dudk = k^4/(12*pi^2)*(3.0*(Ep.^(-1)) .*(coth.(Ep/(2*T))) .+ 2*(Eng.^(-1)) .*(2*sinh.(Eng/T)) .*((cosh.(Eng/T) .- cosh.((2*mu)/T)).^(-1))
			.- 8*(Ek1.^(-1)) .*(1 .- mu*(Eps.^(-1))) .*tanh.(Ek1/(2*T)) .- 8*(Ek2.^(-1)) .*(1 .+ mu*(Eps.^(-1))) .*tanh.(Ek2/(2*T))
			.- 4*(1*Eps.^(-1) .*(tanh.((Eps .- mu)/(2*T)) .+ tanh.((Eps .+ mu)/(2*T))))
           .+ 2*sig_diq_mat_sum.(k, ux, uxx, uy, uyy, uxy, xx, yy, T, mu) )
  return dudk
end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

cutoff = 1000
lam, v, m_lam, hx, hy, c = 0.001, 0, 969/cutoff, 4.2, 3.0, 1750000/cutoff^3
lamy, vy, m_lamy, lamxy = 8, 0, 900/cutoff, -0.6
params = [lam, v, m_lam, lamy, vy, m_lamy, lamxy]
L = 180^2/cutoff^2
L2 = 180^2/cutoff^2
Nx = 60
x = LinRange(0, L, Nx)
Ny = 60
y = LinRange(0, L2, Ny)
dx = abs(x[2] - x[1])
dy = abs(y[2] - y[1])
xgrd, ygrd = meshgrid(x, y)
xx, yy = vcat(xgrd...), vcat(ygrd...)

Nx_new, Ny_new = 8*Nx, 8*Ny
xnew = LinRange(0, L, Nx_new)
ynew = LinRange(0, L2, Ny_new)
dx_new = abs(xnew[2] - xnew[1])
dy_new = abs(ynew[2] - ynew[1])
xgrd_new, ygrd_new = meshgrid(xnew, ynew)
xx_new, yy_new = vcat(xgrd_new...), vcat(ygrd_new...)

N_k = 100 #60
k_IR = 80/cutoff #100/cutoff
k_stop = cutoff/cutoff
global counter = 1
global progress_steps = 500
#k1 = LinRange(cutoff/cutoff, 0.44, Int64(N_k/3))
#k2 = LinRange(0.435, 0.31, Int64(N_k/3))
#k3 = LinRange(0.305, k_IR, Int64(N_k/3))
#k = vcat([k1, k2, k3]...)
k = LinRange(cutoff/cutoff, k_IR, N_k)
print("k grid: ", k)
dk = k[1] - k[2]

function potUV(X, Y, params)
  lam, v, m_lam, lamy, vy, m_lamy, lamxy = params
  return 1/2*m_lam^2*X .+ lam/4*(X .- v^2).^2 .+ 1/2*m_lamy^2*Y .+ lamy/4*(Y .- vy^2).^2 .+ lamxy/4*(X .* Y)
end

#PARALLEL
T_min, T_max = 5/cutoff, 120/cutoff
mu_min, mu_max = 0.1/cutoff, 400/cutoff
N_T, N_mu = 2, 3
T_array, mu_array = LinRange(T_min, T_max, N_T), [mu_min, 310/cutoff, mu_max] #LinRange(mu_min, mu_max, N_mu)
mu_ax, T_ax = meshgrid(T_array, mu_array)
sol = []
min_vals_sig, min_vals_diq, m_sig, m_pi, m_G, m_NG = Array{Union{Nothing, Float64}}(nothing, N_T, N_mu), Array{Union{Nothing, Float64}}(nothing, N_T, N_mu),Array{Union{Nothing, Float64}}(nothing, N_T, N_mu),Array{Union{Nothing, Float64}}(nothing, N_T, N_mu),Array{Union{Nothing, Float64}}(nothing, N_T, N_mu),Array{Union{Nothing, Float64}}(nothing, N_T, N_mu)
tspan = (k[1], k[end])

@time begin
    @inbounds Threads.@threads for i = 1:N_T
        @inbounds Threads.@threads for j = 1:N_mu
            p = [T_array[i], mu_array[j]]
            println("T, mu: ", p .* cutoff)
            prob = ODEProblem(dudk_par, potUV.([xx], [yy], [params])[1], tspan, p, reltol=1e-15, abstol=1e-15)
            s = solve(prob, alg_hints=[:stiff], saveat=k)
            append!(sol, [[T_array[i], mu_array[j], s]])
        end
    end
end

sol = sort(sol)
for i = 1:N_T
    for j = 1:N_mu
        if sol[Int64((i-1)*N_mu + j)][3].t[end] > k_IR
            println("INTEGRATION NOT SUCCESSFUL")
            println(sol[Int64((i-1)*N_mu + j)][3].t[end])
        end
        expl = c* sqrt.(xgrd_new) + 2*mu_array[j]^2*ygrd_new
        spl = Spline2D(xx, yy, sol[Int64((i-1)*N_mu + j)][3].u[end]; kx=3, ky=3, s=0.1)
        sol_interp = vcat(evalgrid(spl, xnew, ynew)...)
        potIR = reshape(sol_interp, Nx_new, Ny_new)
        min_pos = argmin(potIR .- expl)
        fpi, diq = sqrt((min_pos[1]-1)*dx_new)*cutoff, sqrt((min_pos[2]-1)*dy_new)*cutoff
        println("fpi, diq: ", (fpi, diq))
        min_vals_sig[i, j] = fpi
        min_vals_diq[i, j] = diq
    end
end

name = string("QMDhx",string(hx),"hy",string(hy),"mu_min",string(mu_min*cutoff),"Nmu",string(N_mu),"Nk",string(N_k))
mkdir(name)
println(name)

export_arr = vcat([cutoff, v, vy, lam, lamy, m_lam, m_lamy, lamxy, hx, hy, c, Nx, Ny, N_k, N_T, N_mu, x, y, k, T_array, mu_array]...)
writedlm(string(name, "/", "params.csv"), export_arr, ',')
for i = 1:N_T
    export_arr = []
    for j = 1:N_mu
        sol_export = []
        for k = 1:N_k
            append!(sol_export, sol[Int64((i-1)*N_mu + j)][3].u[k])
        end
        append!( export_arr, sol_export )
    end
    writedlm(string(name, "/", i, ".csv"), export_arr, ',')
end
