%initialize parameters
Population = 2000;
%2 percent elites whose gene will not change
EliteRatio = 0.02;
% 70 percent of the population will be eliminated
%also means 28 percent of the population is normal
EliminationRatio = 0.7;
%every gene in chromosome have 1% probability to mutate
MutationRatio = 0.01;
%number of gene in the chromosome
ChromosomeDim = 9;
%extent of mutation
disper = 0.1;
%number of max generation
MaxIter = 300;
%number of MC loop
M = 1000;
%matrix to store result
MCWeight = zeros(ChromosomeDim, M);

%parameters for generating stock prices
T = 20;
%one year has 241 trading days
k = 251 * T;
S0 = 1;
sigma = 0.05;
mu = 0.06;


parfor l = 1:M
    
    %generate first stock price
    St = ones(k + 1, 9);
    % GenerateBS(T, k, sigma, mu, S0)
    % GenerateHeston(T, k, mu, kappa, theta, volvol, S0, rho, v0)
    % GenerateCIR(T, k, sigma, kappa, theta, voldrift, S0, rho, mu0)
    %S1 S2 S3 are BS, S4 S5 S6 are Heston, S7 S8 S9 are CIR-BS
    for j = 0:2
        St(:, 3 * j + 1) = GenerateBS(T, k, sigma + 0.01 * j, mu + 0.01 * j, 1);
        St(:, 3 * j + 2) = GenerateHeston(T, k, mu + 0.01 * j, 0.1, sigma + 0.01 * j, 1, S0, 0.5 - 0.1 * j, sigma + 0.01 * j);
        St(:, 3 * j + 3) = GenerateCIR(T, k, sigma + 0.01 * j, 0.1, mu + 0.01 * j, 1, S0, 0.5 - 0.1 * j, mu + 0.01 * j);
    end
    
    BestStr = zeros(9, MaxIter + 1);
    %initialize the chromosome
    Chromosome = InitChromo(ChromosomeDim, disper, Population);
    %use generated chromosome get the first fitness
    InitFitness = GetFitness(Chromosome, St);
    %sort individuals by fitness
    [InitFitness, idx] = sortrows(InitFitness, 'descend');
    Chromosome = Chromosome(idx);
    %get the most fitted individual
    BestStr(:, 1) = Chromosome{1};
    
    for i = 1:MaxIter
        %generate new stock price
        St = ones(k + 1, 9);
        % GenerateBS(T, k, sigma, mu, S0)
        % GenerateHeston(T, k, mu, kappa, theta, volvol, S0, rho, v0)
        % GenerateCIR(T, k, sigma, kappa, theta, voldrift, S0, rho, mu0)
        for j = 0:2
            St(:, 3 * j + 1) = GenerateBS(T, k, sigma + 0.01 * j, mu + 0.01 * j, 1);
            St(:, 3 * j + 2) = GenerateHeston(T, k, mu + 0.01 * j, 0.1, sigma + 0.01 * j, 1, S0, 0.5 - 0.1 * j, sigma + 0.01 * j);
            St(:, 3 * j + 3) = GenerateCIR(T, k, sigma + 0.01 * j, 0.1, mu + 0.01 * j, 1, S0, 0.5 - 0.1 * j, mu + 0.01 * j);
        end
        %get new generation
        [Fitness, Chromosome] = GetNewGeneration(Chromosome, Population, EliteRatio, EliminationRatio, MutationRatio, St);
        %get most fitted individual in new generation
        BestStr(:, i + 1) = Chromosome{1};
        %display results
        if mod(i, 150) == 0 || i == 0
            disp(['for ', num2str(i), '-th loop, Max fitness is ', num2str(Fitness(1)), ', Mean Fitness is ', num2str(nanmean(Fitness))])
            disp('Best fitted weight is')
            disp(Chromosome{1}')
        end
    end
    
    %plots
    for i = 1:9
        plot(BestStr(i, :), 'o')
        hold on
    end
    hold off
    MCWeight(:, l) = Chromosome{1};
    disp(l)
end



function [Fitness, New_Chromosome] = GetNewGeneration(Chromosome, Population, EliteRatio, EliminationRatio, MutationRatio, St)

Fitness = GetFitness(Chromosome, St);

%sort by fitness
[Fitness, idx] = sortrows(Fitness, 'descend');
New_Chromosome = Chromosome(idx);

%determine the number of elite and normal individual
Start = floor(EliteRatio * Population);
Stop = Population - floor(EliminationRatio * Population);

%keep elite gene unchanged
%the normal individuals crossover with themselves
for i = Start:2:Stop
    [New_Chromosome{i}, New_Chromosome{i + 1}] = Crossover(New_Chromosome{i}, New_Chromosome{i + 1}, MutationRatio);
end

%remove all of the bad gene
%use crossover of left gene to fulfill their niches
%it is possible for a individual to crossover itself, such as parthenogenesis
for i = (Stop + 2):Population
    New_Chromosome{i} = Crossover(New_Chromosome{randi(Stop)}, New_Chromosome{randi(Stop)}, MutationRatio);
end

end

function chromosome = InitChromo(InputSize, sigma, Population)
    %only generate the first n-1 dimension, because sum(weight) = 1
    parfor i = 1:Population
        Matrix = randn(InputSize - 1, 1) * sigma;
        temp = 1 - sum(Matrix);
        chromosome{i} = [Matrix; temp];
    end
end

function New_Chromosome = Mutation(Chromosome, MutationRatio)

%when th i-th element of id is 1, the the i-th gene will mutate
id = binornd(1,MutationRatio, [length(Chromosome) - 1, 1]);

%if mutated, the new weight is St = St * (1 + N), N is a random number
Chromosome(1:end - 1) = Chromosome(1:end - 1) .* ( 1 + randn(length(Chromosome) - 1, 1).* id);

New_Chromosome = [Chromosome(1:end - 1); 1 - sum(Chromosome)];

end

function [Child1, Child2] = SwapChromosome(Father, Mother, point)
    Child1 = Father;
    Child2 = Mother;
    
    %swap the tail of father and mother
    Child1(1 : point) = Mother(1 : point);
    Child2(1 : point) = Father(1 : point);
end

function [Child1, Child2, point] = Crossover(Father, Mother, MutationRatio)
    RoundedFather = Father(1 : end - 1);
    RoundedMother = Mother(1 : end - 1);
    
    %choose the point to swap chromosome
    point = randi(length(RoundedFather) - 1);

    [Child1, Child2] = SwapChromosome(RoundedFather, RoundedMother, point);
    
    %mutation
    Child1 = Mutation(Child1, MutationRatio);
    Child2 = Mutation(Child2, MutationRatio);
    
    %determine the last dimension of weight
    temp = 1 - sum(Child1);
    Child1 = [Child1; temp];
    temp = 1 - sum(Child2);
    Child2 = [Child2; temp];
end

function Fitness = GetFitness(Chromosome, St)
    
    %determine paramenters
    penalty = 0.01;
    w = 0.3;
    
    for i = 1:length(Chromosome)
        %get portfolio value vs time
        Portfolio =  St * Chromosome{i};
        %take rate of return, sd, max dropdown and lower point as our
        %indexs
        [RoR, SD, MaxDropdown, lowerpoint] = GetPara(1, Portfolio);
        
        if RoR > 0
            %we set risk free rate = 0 so RoR / SD is sharpe ratio
            %if there is a shorted stock, there is 1% penalty in fitness
            Fitness(i) = RoR / SD - w * MaxDropdown + (1 - w) * lowerpoint - penalty * sum(Chromosome{i} < 0);
        else 
            Fitness(i) = RoR * SD - w * MaxDropdown + (1 - w) * lowerpoint - penalty * sum(Chromosome{i} < 0);
        end
    end
    
    %for multiply of matrix
    Fitness = Fitness';
    
end

%we use Euler's scheme to generate stock price
function St = GenerateBS(T, k, sigma, mu, S0)

dt = T / k;
dWt = randn(k, 1) * sqrt(dt);
St = S0 * ones(1, 1 + k);

for i = 1:k
    St(i + 1) = St(i) + St(i) * mu * dt + St(i) * sigma *dWt(i);
end

end

function St = GenerateHeston(T, k, mu, kappa, theta, volvol, S0, rho, v0)

dt = T / k;
dWst = randn(k, 1) * sqrt(dt);
dWvt = sqrt(1 - rho^2) * randn(k, 1) * sqrt(dt) + rho * dWst;

St = S0 * ones(1, 1 + k);
vt = v0 * ones(1, 1 + k);

for i = 1:k
    vt(i + 1) = max(0, vt(i) + kappa * (theta - vt(i)) * dt + volvol * sqrt(vt(i)) * dWvt(i));
    St(i + 1) = St(i) + St(i) * mu * dt + St(i) * sqrt(vt(i)) *dWst(i);
end

end

function St = GenerateCIR(T, k, sigma, kappa, theta, voldrift, S0, rho, mu0)

dt = T / k;
dWst = randn(k, 1) * sqrt(dt);
dWmut = sqrt(1 - rho^2) * randn(k, 1) * sqrt(dt) + rho * dWst;

St = S0 * ones(1, 1 + k);
mut = mu0 * ones(1, 1 + k);

for i = 1:k
    mut(i + 1) = max(0, mut(i) + kappa * (theta - mut(i)) * dt + voldrift * sqrt(mut(i)) * dWmut(i));
    St(i + 1) = St(i) + St(i) * mut(i) * dt + St(i) * sigma *dWst(i);
end

end    
    
function [RoR, SD, MaxDropdown, lowerpoint] = GetPara(T, St)

%use log return
logSt = log(St);
RoR = (logSt(end) - logSt(1)) / T;
SD = sqrt(var(logSt) * length(St) / T);

for i = 1:T
    yrdrop = - min(diff(logSt((251 * (i-1) + 1):251 * i)));
    MaxDropdown(i) = max(0, yrdrop);
    yrlower = min(logSt((251 * (i-1) + 1):251 * i) - logSt(251 * (i-1) + 1));
    lowerpoint(i) = min(0, yrlower);
end

%mean of every year's data
MaxDropdown = mean(MaxDropdown);
lowerpoint = mean(lowerpoint);

%if the portfolio value is negative, it goes to bankcrupcy (too high leverage)
if sum(St < 0) > 0
    RoR = -inf; SD = 1; MaxDropdown = 0; lowerpoint = 0;
end

end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    