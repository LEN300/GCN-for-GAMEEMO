import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score

# 定义一个函数，用于计算节点之间的互信息量
def mutual_information(x,y):
    # x和y是两个一维数组，表示两个节点的信号或活动
    # 返回x和y之间的互信息量
    return mutual_info_score(x,y)

# 定义一个函数，用于计算节点在网络中的信息效率
def information_efficiency(G,i):
    # G是一个networkx图对象，表示脑网络矩阵
    # i是一个整数，表示要计算信息效率的节点编号（从0开始）
    # 返回节点i在网络G中的信息效率值

    # 获取网络中节点总数
    N = G.number_of_nodes()

    # 初始化信息效率为0
    E_i = 0

    # 遍历其他所有节点j（除了i本身）
    for j in range(N):
        if j != i:
            # 获取节点i和j之间传递信息的互信息量（假设G有一个属性叫signal，存储每个节点对应信号或活动）
            I_ij = mutual_information(G.nodes[i]['signal'],G.nodes[j]['signal'])

            # 获取节点i和j之间最短路径长度（如果不存在路径，则返回无穷大）
            d_ij = nx.shortest_path_length(G,i,j)

            # 累加信息效率分子部分（如果d_ij为无穷大，则跳过该项）
            if d_ij != np.inf:
                E_i += I_ij / d_ij

    # 计算并返回信息效率值（除以N-1）
    E_i = E_i / (N-1)
    return E_i

# 定义一个函数，用于确定合适的阈值
def optimal_threshold(G):
    # G是一个networkx图对象，表示脑网络矩阵
    # 返回使得全局效率达到最大值时对应的阈值

    # 获取网络中节点总数
    N = G.number_of_nodes()

    # 计算所有节点在原始网络中的信息效率，并存储在一个列表中（按照编号顺序）
    E_list = []
    for i in range(N):
        E_list.append(information_efficiency(G,i))

    # 将列表转换为numpy数组，并排序（从小到大）
    E_array = np.array(E_list)
    E_array.sort()

    # 初始化最优阈值为0，最高全局效率为0
    theta_E = 0
    global_efficiency_max = 0

         
    for theta in E_array:

#遍历所有可能的阈值（从最小到最大），并计算相应过滤后网络的全局效率；选择使得全局效率达到最大值时对应的阈值作为最终阈值。
#注意：这里我们假设E_array没有重复元素；如果有重复元素，则需要先去重或者跳过重复元素。
        # 根据阈值对原始网络进行过滤，即保留那些信息效率大于或等于theta 的节点及其连接，并删除那些信息效率小于theta 的节点及其连接。
        H = G.copy()  #删除H中不符合条件的节点及其连接
        for i in range(N):
            if E_list[i] < theta:
                # 删除节点i及其连接
                H.remove_node(i)
        # 计算过滤后网络的全局效率
        global_efficiency = nx.global_efficiency(H)

        # 如果全局效率大于当前最高值，则更新最优阈值和最高全局效率
        if global_efficiency > global_efficiency_max:
            theta_E = theta
            global_efficiency_max = global_efficiency

    # 返回最优阈值
    return theta_E

# 定义一个函数，用于对脑网络进行过滤
def filter_brain_network(G):
    # G是一个networkx图对象，表示脑网络矩阵
    # 返回一个过滤后的networkx图对象，表示更真实和有用的脑网络

    # 获取网络中节点总数
    N = G.number_of_nodes()

    # 计算合适的阈值
    theta_E = optimal_threshold(G)

    # 根据阈值对原始网络进行过滤，即保留那些信息效率大于或等于theta_E 的节点及其连接，并删除那些信息效率小于theta_E 的节点及其连接。
    H = G.copy()  #删除H中不符合条件的节点及其连接
    
    for i in range(N):
        if information_efficiency(G,i) < theta_E:
            # 删除节点i及其连接
            H.remove_node(i)

    # 返回过滤后的网络
    return H

# 假设我们已经有了一个表示脑网络矩阵的networkx图对象G（这里省略了如何构建G的代码）
# 调用上面定义的函数，对G进行过滤，并得到一个新的networkx图对象H（表示更真实和有用的脑网络）
H = filter_brain_network(G)