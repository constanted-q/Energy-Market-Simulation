# Energy-Market-Simulation
For the setting:
1. Capacity, i.e. the maximum amount of energy that can be held is set to be 7
2. Every client has his own G, such that he uniformly randomly generates 0 to (G-1) units of energy each day.
3. At the beginning of the day, each client observes state s, the amount of energy remaining; choose a, the amount of energy to consume; then ask for {max(0,a-s)} units of energy from the market; the client receives {r = a^2-max(0,a-s)*price} as the reward; at the end of the day s' become {max(0,s-a)+g}, where g is from 2. The rewards are devided by capacity^2 which is the max possible reward.
4. The price of energy per unit is determined by {lambda1*sum(asking)}, where sum(asking) is the sum of the amount of energy every client is asking for from the market. lambda1 is set to be 1.5 (and 0 is the "free energy" case).
5. Other parameters in Alg 1 of paper: delta = 1/20*A*S = 1/20*(capacity+1)^2. Stepsize eta = 0.02. K = 2000. d = 500. h(x) = 1/2 x^2.
