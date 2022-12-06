from WirelessEnvRL import WirelessEnv
import numpy as np

env = WirelessEnv()


#print(env.step_no,state[1])

done = False
#while done==False:
action_reward = []
action_bcost = []
action_pcost = []
action_len = []
for action in range(3,18):
    full_reward = []
    full_bcost = []
    full_pcost = []
    full_len = []
    for i in range(1):
        done=False
        state = env.reset()
        reward_sum = 0
        bcost_sum = 0
        pcost_sum = 0
        episode_len = 0
        for k in range(200):
            next_state,reward,done,info = env.step(action) 
            reward_sum += reward
            bcost_sum += info["Buffer Cost"]
            pcost_sum += info["Power Cost"]
            episode_len += 1
        full_reward += [reward_sum]
        full_bcost += [bcost_sum]
        full_pcost += [pcost_sum]
        full_len += [episode_len]
    action_reward += [full_reward]
    action_bcost += [full_bcost]
    action_pcost += [full_pcost]
    action_len += [full_len]
    print("*************************************")
    print("Power - ",env.agent_Power," Modulation -", env.agent_M," Avg length of episodes - ", np.mean(np.array(full_len)))
    print("Reward Full - ",np.sum(np.array(full_reward))," Buffer Cost - ",np.sum(np.array(full_bcost))," Power Cost - ", np.sum(np.array(full_pcost)) )
    print("Avg Reward Full - ",np.mean(np.array(full_reward)/np.array(full_len)),"Avg Buffer Cost - ",np.mean(np.array(full_bcost)/np.array(full_len)),"Avg Power Cost - ", np.mean(np.array(full_pcost)/np.array(full_len)) )
    print("*************************************")
action_reward = np.array(full_reward)
action_bcost = np.array(full_bcost)
action_pcost = np.array(full_pcost)
action_len += np.array(action_len)

np.save("Results/Action_reward.npy",action_reward)
np.save("Results/Action_bcost.npy",action_bcost)
np.save("Results/Action_pcost.npy",action_pcost)
np.save("Results/Action_len.npy",action_len)

