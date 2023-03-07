from pylab import mpl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_ann(number_input, number_hidden, number_output):
    """
    number_input:输入层节点个数
    number_hidden:隐藏层各层节点个数
    number_output:输出层节点个数
    """
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # matplotlib使用中文，SimHei为黑体
    ceng_hidden = len(number_hidden)  # 隐藏层层数
    g = nx.DiGraph()

    # 节点
    vertex_input_list = ['v' + str(i) for i in range(1, number_input + 1)]  # 输入层
    vertex_hidden_list = []
    start = number_input + 1
    end = number_input + number_hidden[0] + 1
    vertex_hidden_list.append(['v' + str(i) for i in range(start, end)])  # 隐藏层
    for j in range(1, ceng_hidden):
        start = end
        end = start + number_hidden[j]
        vertex_hidden_list.append(['v' + str(i) for i in range(start, end)])  # 隐藏层
    vertex_output_list = ['v' + str(i) for i in range(end, end + number_output)]  # 输出层
    vertex_list = []
    vertex_list.extend(vertex_input_list)
    list(map(lambda i: vertex_list.extend(vertex_hidden_list[i]), range(ceng_hidden)))
    vertex_list.extend(vertex_output_list)
    g.add_nodes_from(vertex_list)

    # 连接
    edge_input_hidden_list = []
    edge_input_hidden_list.extend([(i, j) for i in vertex_input_list for j in vertex_hidden_list[0]])  # 输入层-隐藏层
    edge_list = []
    edge_list.extend(edge_input_hidden_list)
    edge_hidden_hidden_list = []
    if ceng_hidden > 1:
        for k in range(ceng_hidden - 1):
            edge_hidden_hidden_list.extend(
                [(i, j) for i in vertex_hidden_list[k] for j in vertex_hidden_list[k + 1]])  # 隐藏层-隐藏层
        edge_list.extend(edge_hidden_hidden_list)
    edge_hidden_output_list = []
    edge_hidden_output_list.extend(
        [(i, j) for i in vertex_hidden_list[len(vertex_hidden_list) - 1] for j in vertex_output_list])  # 隐藏层-输出层
    edge_list.extend(edge_hidden_output_list)
    g.add_edges_from(edge_list)

    # 位置
    pos = {}
    ceng_pos_x = np.linspace(-(ceng_hidden + 2) / 2, (ceng_hidden + 2) / 2, num=ceng_hidden + 2)
    list(map(lambda i: pos.update({vertex_input_list[int(np.where(np.arange(
        -number_input / 2 * 1 + 1 / 2, number_input / 2 * 1 + 1 / 2, 1) == i)[0])]: (ceng_pos_x[0], i)}),
             np.arange(-number_input / 2 * 1 + 1 / 2, number_input / 2 * 1 + 1 / 2, 1)))  # 输入层
    list(map(lambda j: list(map(lambda i: pos.update({vertex_hidden_list[j][int(np.where(np.arange(
        -number_hidden[j] / 2 * 1 + 1 / 2, number_hidden[j] / 2 * 1 + 1 / 2, 1) == i)[0])]: (ceng_pos_x[j + 1], i)}),
                                np.arange(-number_hidden[j] / 2 * 1 + 1 / 2, number_hidden[j] / 2 * 1 + 1 / 2, 1))),
             range(ceng_hidden)))  # 隐藏层
    list(map(lambda i: pos.update({vertex_output_list[int(np.where(np.arange(
        -number_output / 2 * 1 + 1 / 2, number_output / 2 * 1 + 1 / 2, 1) == i)[0])]: (
    ceng_pos_x[len(ceng_pos_x) - 1], i)}),
             np.arange(-number_output / 2 * 1 + 1 / 2, number_output / 2 * 1 + 1 / 2, 1)))  # 输出层

    fig = plt.figure(figsize=(8, 5), dpi=300)
    plt.xlim(ceng_pos_x[0] - 1, ceng_pos_x[len(ceng_pos_x) - 1] + 1)
    plt.ylim(-max(number_input, max(number_hidden), number_output) / 2 * 1,
             max(number_input, max(number_hidden), number_output) / 2 * 1 + 1 / 2)

    nx.draw(
        g,
        pos=pos,
        node_color='blue',
        edge_color='black',
        with_labels=False,
        font_size=10,
        node_size=300,
    )
    fig.savefig('全连接层网络可视化.png')


if __name__ == '__main__':
    plot_ann(3, [4,4,4], 2)
