# 基于micROS机器人操作系统典型应用集成与验证

> 姓名：马骁腾
>
> 班级：自博171
>
> 学号：2017310831
>
> 实践单位：天津（滨海）人工智能军民融合创新中心
>
> 项目名称：基于micROS机器人操作系统的典型应用集成与验证

本课题内容为基于micROS机器人操作系统的典型应用集成与验证，要求围绕实践单位现有的micROS操作系统上进行相关应用的开发。本人分配的任务是开发分布式机器学习训练系统并验证多智能体强化学习算法。实践的主要内容和完成工作包括：

- Kubernetes集群支持
  - 安装Nvidia-Docker
  - 配置kubeadm，使用DevicePlugins支持GPU使用
  - 在K8s集群中启用GPU支持
- 在Kubernetes上部署Ray
  - 封装Docker镜像：以Nvidia官方镜像为基础，封装Tensorflow、PyTorch等依赖，最后封装Ray
  - 创建Ray的Service，ray-head和ray-worker
  - 创建使用NodePort的Service，通过Jupyter Notebook和Tensorboard来访问Ray集群
- 使用Ray实现MADDPG算法
  - 创建MultiAgentEnv，将状态、动作、奖励等信息以字典的形式返回
  - 实现MADDPGPolicy，通过postprocess_trajectory实现不同智能体间的信息共享用于训练
  - 基于GenericOffPolicyTrainer创建一个MADDPGTrainer来训练算法

以上工作将在本文后面部分中详细地展开介绍技术方案和要点。本文的组织结构如下：

[TOC]



## Kubernetes

Kubernetes，简称K8s，是用8代替8个字符“ubernete”而成的缩写。是一个开源的，用于管理云平台中多个主机上的容器化的应用，Kubernetes的目标是让部署容器化的应用简单并且高效，Kubernetes提供了应用部署，规划，更新，维护的一种机制。

Kubernetes的基本应用不支持使用GPU的机器学习任务，但在近期的版本中使用了DevicePlugins的形式为用户提供调用GPU的接口。

### 安装Nvidia-Docker
参考[github nvidia-docker](https://github.com/NVIDIA/nvidia-docker)，如果docker版本不一致可以参考
[nvidia-docker wiki](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-20-if-im-not-using-the-latest-docker-version)搜索需要的版本
```bash
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image 
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

### 配置docker daemon config

检查每一个节点，启用 nvidia runtime为缺省的容器运行时。我们将编辑docker daemon config文件，位于/etc/docker/daemon.json
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### 配置kubeadm
启用 DevicePlugins feature gate，在每一个GPU节点都要设置。如果你的 Kubernetes cluster是通过kubeadm部署的，并且节点运行systemd，需要打开kubeadm 的systemd unit文件，位于`/etc/systemd/system/kubelet.service.d/10-kubeadm.conf` 然后添加下面的参数作为环境变量：`Environment="KUBELET_GPU_ARGS=--feature-gates=DevicePlugins=true"`

```bash
# Note: This dropin only works with kubeadm and kubelet v1.11+

[Service]
Environment="KUBELET_KUBECONFIG_ARGS=--bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf 
--kubeconfig=/etc/kubernetes/kubelet.conf"
Environment="KUBELET_CONFIG_ARGS=--config=/var/lib/kubelet/config.yaml"
Environment="KUBELET_GPU_ARGS=--feature-gates=DevicePlugins=true"

EnvironmentFile=-/var/lib/kubelet/kubeadm-flags.env
EnvironmentFile=-/etc/sysconfig/kubelet

ExecStart=
ExecStart=/usr/bin/kubelet $KUBELET_KUBECONFIG_ARGS 
$KUBELET_CONFIG_ARGS $KUBELET_KUBEADM_ARGS 
$KUBELET_EXTRA_ARGS $KUBELET_GPU_ARGS
```

配置完成后重启kubelet服务
```bash
sudo systemctl daemon-reload
sudo systemctl restart kubelet
```

### 在Kubernetes中启用GPU支持
完成所有的GPU节点的选项启用，然后就可以在在Kubernetes中启用GPU支持，通过安装Nvidia提供的Daemonset服务来实现，方法如下，**注意修改版本**：
`kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.10/nvidia-device-plugin.yml`
注：集群上只需要配置一次，如果其中一个节点已经配置成功，则其他节点不需要再配置。
完成上述工作后**重启**。

### 测试
获取当前节点的GPU资源
```bash
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```
配置测试pod：`gpu-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  restartPolicy: Never
  containers:
  - image: nvidia/cuda
    name: cuda
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```
```bash
kubectl create -f gpu-pod.yaml
kubectl logs gpu-pod
```
### Docker镜像
完成上述步骤后k8s已经可以找到GPU资源，但还没有nvidia的相关环境，需要将相关配置封在docker里，在Dockerfile里加上`FROM nvidia/cuda:10.1-runtime-ubuntu16.04`，注意修改cuda版本标签

### 参考资料
* https://blog.csdn.net/u013531940/article/details/79674792
* https://blog.csdn.net/xingyuzhe/article/details/81908701
* https://my.oschina.net/u/2306127/blog/1808304
* https://github.com/NVIDIA/nvidia-docker
* 
https://hub.docker.com/r/nvidia/cuda/

## Ray

Ray是UCB研发的开源分布式计算框架，主要包括自动参数调整Tune和强化学习框架RLlib两部分。


### Tune
Tune是Ray提供的用于调整超参数的工具，提供了多种参数搜索算法。即使无需调整超参数，也可以作为训练程序的入口，优势在于更好的参数配置、日志输出、结果展示。RLlib里面的算法可以被封装为Trainer被Tune调用。
下面是使用Tune训练CartPole并搜索学习率的示例：

```python
import ray
from ray import tune

ray.init(redis_address="localhost:6379")
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
)
```
* * *
![Tune逻辑图](https://ray.readthedocs.io/en/latest/_images/tune-api.svg)
Tune接受两种不同的方式进行训练，function-based API或Trainable API。定义function-based API的方式如下：
```python
def trainable(config, reporter):
    """    
    Args:        
        config (dict): Parameters provided from the search algorithm            
            or variant generation.       
        reporter (Reporter): Handle to report intermediate metrics to Tune.    
    """

    while True:
        # ...
        reporter(**kwargs)
```
其中reporter属于类`ray.tune.function_runner.StatusReporter`，记录了用于控制参数搜索算法的性能反馈，如mean_accuracy。除了使用reporter，也可以用track.log（from ray.tune import track）。
Trainable API则需要定义一个类`Trainable`，子类需要覆写`_setup`、`_test`、`_train`、`_save`、`_restore`等函数。
可以参考使用PyTorch训练MNIST类似的来实现自己的Tune可训练对象，分别见[mnist_pytorch](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch.py)和[mnist_pytorch_trainable](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch_trainable.py)。

### RLlib

RLlib是运行在Ray上层的强化学习框架，提供了丰富的接口函数用于快速构建算法。
![RLlib层次概念图](https://ray.readthedocs.io/en/latest/_images/rllib-stack.svg)

#### Environments
RLlib支持OpenAI Gym作为仿真环境，也可以支持用户自定义的环境。下面重点讲一下如何定义多智能体的仿真环境。

##### MultiAgentEnv
Ray定义了类`MultiAgentEnv`来处理多智能体的环境，将原gym函数返回值拓展为了{agent_id: value}的形式
以下是example中的示例，全部代码在[twostep_game](https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/twostep_game.py)：
```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class TwoStepGame(MultiAgentEnv):
    action_space = Discrete(2)
    # Each agent gets a separate [3] obs space, to ensure that they can
    # learn meaningfully different Q values even with a shared Q model.
    observation_space = Discrete(6)
    def __init__(self, env_config):
        self.state = None
    def reset(self):
        self.state = 0
        return {"agent_1": self.state, "agent_2": self.state + 3}
    def step(self, action_dict):
        if self.state == 0:
            action = action_dict["agent_1"]
            assert action in [0, 1], action
            if action == 0:
                self.state = 1
            else:
                self.state = 2
            global_rew = 0
            done = False
        elif self.state == 1:
            # ...
        rewards = {"agent_1": global_rew / 2.0, "agent_2": global_rew / 2.0}
        obs = {"agent_1": self.state, "agent_2": self.state + 3}
        dones = {"__all__": done}
        infos = {}
        return obs, rewards, dones, infos
```
通过`Policy`中的`postprocess_trajectory`可以处理共享信息的问题，利用`other_agent_batches`和`episode`来获取其他agent的状态，例如：
```python
def postprocess_trajectory(policy, sample_batch, other_agent_batches, episode):
    agents = ["agent_1", "agent_2", "agent_3"]  # simple example of 3 agents
    global_obs_batch = np.stack(
        [other_agent_batches[agent_id][1]["obs"] for agent_id in agents],
        axis=1)
    # add the global obs and global critic value
    sample_batch["global_obs"] = global_obs_batch
    sample_batch["central_vf"] = self.sess.run(
        self.critic_network, feed_dict={"obs": global_obs_batch})
    return sample_batch
```

#### Models and Preprocessors
##### Model
RLlib支持使用内建模型定义和自定义模型两种方式。如果你的问题只需要使用FC或者CNN的话，可以直接在config中说明需要模型的参数即可。例如：
```python
"model": {"fcnet_activation": "tanh", "fcnet_hiddens": [64, 64]}    # FC
"model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]}    # CNN
```
如果需要使用lstm的话只需要添加`use_lstm=True`就可以了。

内建模型对输入输出的局限比较大，这时候需要用自定义模型。自定义模型也很容易，只需要复写类`Model`（TensorFlow）和`TorchModel`（PyTorch）。
```python
import ray
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.pytorch.model import TorchModel

class CustomTorchModel(TorchModel):

    def __init__(self, obs_space, num_outputs, options):
        TorchModel.__init__(self, obs_space, num_outputs, options)
        ...  # setup hidden layers

    def _forward(self, input_dict, hidden_state):
        """Forward pass for the model.
        
        Prefer implementing this instead of forward() directly for proper
        handling of Dict and Tuple observations.
        
        Arguments:
            input_dict (dict): Dictionary of tensor inputs, commonly
                including "obs", "prev_action", "prev_reward", each of shape
                [BATCH_SIZE, ...].
            hidden_state (list): List of hidden state tensors, each of shape
                [BATCH_SIZE, h_size].
                
        Returns:
            (outputs, feature_layer, values, state): Tensors of size
                [BATCH_SIZE, num_outputs], [BATCH_SIZE, desired_feature_size],
                [BATCH_SIZE], and [len(hidden_state), BATCH_SIZE, h_size].
        """
        obs = input_dict["obs"]
        ...
        return logits, features, value, hidden_state

ModelCatalog.register_custom_model("my_model", CustomTorchModel)

ray.init()
trainer = a3c.A2CTrainer(env="CartPole-v0", config={
    "use_pytorch": True,
    "model": {
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
    },
})
```
##### Preprocessor
Preprocessor的作用类似于Gym中的Wrapper。RLlib会自动选择一些内建的Preprocessor，Discrete会被One-Hot编码，Atari会被downscale，Tuple和Dict会被展开。
自定义一个Preprocessor也很容易：
```python

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models.preprocessors import Preprocessor

class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return new_shape  # can vary depending on inputs

    def transform(self, observation):
        return ...  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "model": {
        "custom_preprocessor": "my_prep",
        "custom_options": {},  # extra options to pass to your preprocessor
    },})
```

#### Concepts and Custom Algorithms
![RLlib各模块关系](https://ray.readthedocs.io/en/latest/_images/rllib-components.svg)
        
##### Policy
在Ray实现自定义的算法有两种方式，一种是采用模板`tf_policy_template.build_tf_policy()`和`torch_policy_template.build_torch_policy`，一种是复写类`CustomPolicy`，两种方式本质是一样的，都需要实现一些必要的函数。下面是类`CustomPolicy`的定义，针对TensorFlow和PyTorch优化的模板在[tf_policy](https://github.com/ray-project/ray/blob/master/python/ray/rllib/policy/tf_policy.py)和[torch_policy](https://github.com/ray-project/ray/blob/master/python/ray/rllib/policy/torch_policy.py)：

```python
class CustomPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and    `build_torch_policy` helpers instead for a real policy, which are described in the next sections.    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # example parameter
        self.w = 1.0

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]
```

##### SampleBatch
为了除了复杂的采样信息，Ray定义了类`SampleBatch`，用来储存仿真得到经验和从buffer中采的样本。以下是`SampleBatch`定义的主要信息，其他信息也可以由用户自己添加：
```python
@PublicAPI
class SampleBatch(object):
    """Wrapper around a dictionary with string keys and array-like 
values.
    For example, {"obs": [1, 2, 3], "reward": [0, -1, 1]} is a batch of 
three
    samples, each with an "obs" and "reward" attribute.
    """
    # Outputs from interacting with the environment
    CUR_OBS = "obs"
    NEXT_OBS = "new_obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    PREV_ACTIONS = "prev_actions"
    PREV_REWARDS = "prev_rewards"
    DONES = "dones"
    INFOS = "infos"
    # Uniquely identifies an episode
    EPS_ID = "eps_id"
    # Uniquely identifies a sample batch. This is important to distinguish RNN
    # sequences from the same episode when multiple sample batches are
    # concatenated (fusing sequences across batches can be unsafe).
    UNROLL_ID = "unroll_id"
    # Uniquely identifies an agent within an episode
    AGENT_INDEX = "agent_index"
    # Value function predictions emitted by the behaviour policy
    VF_PREDS = "vf_preds"
```

##### Trainer
只实现Policy是无法完成训练的，训练的算法被打包为Trainer，以便被tune调用。从头实现Trainer比较复杂，RLlib已经实现了一些通用的Trainer模板，可以通过`build_trainer`来实现。
以下是`GenericOffPolicyTrainer`和在此基础上实现的DQN，注意其中实现了大量的训练处理函数，全部代码见[dqn](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/dqn/dqn.py)：
```python
GenericOffPolicyTrainer = build_trainer(
    name="GenericOffPolicyAlgorithm",
    default_policy=None,
    default_config=DEFAULT_CONFIG,
    validate_config=check_config_and_setup_param_noise,
    get_initial_state=get_initial_state,
    make_policy_optimizer=make_optimizer,
    before_init=setup_exploration,
    before_train_step=update_worker_explorations,
    after_optimizer_step=update_target_if_needed,
    after_train_result=add_trainer_metrics,
    collect_metrics_fn=collect_metrics,
    before_evaluate_fn=disable_exploration)
    
DQNTrainer = GenericOffPolicyTrainer.with_updates(
    name="DQN", default_policy=DQNTFPolicy, 
default_config=DEFAULT_CONFIG)
```

### 参考资料
- https://ray.readthedocs.io/en/releases-0.7.2
- https://github.com/ray-project/ray/

## 成果展示

实践的主要内容围绕搭建仿真训练平台开展，这里

https://github.com/xtma/ray-maddpg