MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [
            ['$', '$'],
            ['\\(', '\\)']
        ]
    },
    CommonHTML: {
        linebreaks: {
            automatic: true
        }
    },
    "HTML-CSS": {
        linebreaks: {
            automatic: true
        }
    },
    SVG: {
        linebreaks: {
            automatic: true
        }
    }
});

/*
content: 
`When $a \\ne 0$, there are two solutions to \\(ax^2 + bx + c = 0\\) and they are
        $$x = {-b \\pm \\sqrt{b^2-4ac} \\over 2a}.$$` */

var list = [{
        term: 'Supervised learning',
        link: `supervised-learning`,
        content: /*html*/ `In supervised learning settings, you have training data which also contains desired solutions, so called labels (or predictors, or attributes).
        You use these to train the model how certain features lead to an outcome.`,
        tags: ['ml', 'dl']
    },
    {
        term: 'Unsupervised learning',
        link: `unsupervised-learning`,
        content: /*html*/ `Unsupervised learnig is learning without a teacher. Your training data does not contain desired solutions. Hence, unsupervised learning is particularly good
        for anomaly detection, or finding relations in big, unstructured data.`,
        tags: ['ml', 'dl']
    },
    {
        term: 'Semi-Supervised learning',
        link: `semi-supervised-learning`,
        content: /*html*/ `In semi-supervised settings you have partially labeld data, of which the majority is unlabeled. It is a combination of supervised and supervised learning. 
        Examples are Deep Belief Networks (DBN), consisting of multiple Restricted Boltzmann Machines (RBM).`,
        tags: ['ml', 'dl']
    },
    {
        term: 'Reinforcement learning',
        link: `reinforcement-learning`,
        content: /*html*/ `In reinforcement learning you have an agent that can observe an environment and selects and performs actions, and gets rewards or penalties.
        The agent learns by itself what the best strategy is. This is called Policy. The policy decides which action an agent takes in certain situations.`,
        tags: []
    },
    // {
    //     term: 'Batch learning',
    //     link: `batch-learning`,
    //     content: /*html*/  ``,
    //     tags: []
    // },
    {
        term: 'Online learning',
        link: `online-learning`,
        content: /*html*/ `In online learning a system is fed incrementally (on-the-fly) with data (mini-batches). Advantages are that the system can be kept up-to-date easily, and the 
        data used for training can be thrown away fast.`,
        tags: []
    },
    {
        term: 'Out-of-core learning',
        link: `#out-of-core-learning`,
        content: /*html*/ `Out-of-core learning is learning from data that does not entirely fit into memory, and hence requires some kind of streaming or batching.`,
        tags: []
    },
    {
        term: 'Axis',
        link: `numpy-axis`,
        content: /*html*/ `<p>Axis are used when iterating over a 2D array.</p>
        <pre><code class="python">a = np.array([[1,2,3], [4,5,6]])
> array([[1, 2, 3],
         [4, 5, 6]])</code></pre>
         <p>Numpy <kbd>axis=0</kbd> iterates vertically downwards, column by column<br>
         Numpy <kbd>axis=1</kbd> iterates horizontally left to right, row by row</p>
         <pre><code class="python">a.sum(axis=0)
> array([5, 7, 9])

a.sum(axis=1)
> array([6, 15])</code></pre>`,
        tags: []
    },
    {
        term: 'Numpy Array <i>(row-major)</i>',
        link: `numpy-array`,
        content: /*html*/ `<p>Numpy arrays are by default <i>row-major</i>. This means that every array of values, when creating a numpy.arrary, is a row.</p>
<pre><code class="class python">np.array([[1,2],[3,4]])
> array([[1, 2],
         [3, 4]])
</code></pre><p>
[1, 2] are a row, [1, 3] are a column</p>`,
        tags: []
    },
    {
        term: 'Multiclass Classification',
        link: `multiclass-classification`,
        content: /*html*/ `<p>A classifer that can assign multiple classes to a sample, but never more than one at a time.</p>`,
        tags: ['knn']
    },
    {
        term: 'Multilabel Classification',
        link: `multilabel-classification`,
        content: /*html*/ `<p>A classifer that can assign each sample a set of multiple target labels. For example, a system that detects that a number is odd and greater than 70.
        Given an input of 13, the output could be [1, 0].</p>`,
        tags: ['knn']
    },
    {
        term: 'Multioutput-multiclass Classification',
        link: `multioutput-multiclass-classification`,
        content: /*html*/ `<p>A classifier that can assign multiple classes to an input. The number of assigned classes can vary from sample to sample.
         It is a generalization of the <a href="#multilabel-classification">Multilabel-Classifier</a>.</p>`,
        tags: []
    },
    {
        term: 'Label',
        link: `label`,
        content: /*html*/ `<p>In machine learning a label is the desired output, result, or assigned class based on certrain attribute values (features). 
        The machine learning model should learn to predict based on the attribute values (features) what the desired output is.</p>`,
        tags: []
    },
    {
        term: 'Interquartile Range (IQR)',
        link: ``,
        content: /*html*/ `<p>IQR is a trimmed estimator. It is the midspread of a sample and a meassure of variability.</p>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/598px-Boxplot_vs_PDF.svg.png" alt="">
        <p><small><a href="https://en.wikipedia.org/wiki/User:Jhguch">Jhguch</a> at <a href="http://en.wikipedia.org">en.wikipedia</a>,
        <a href="https://commons.wikimedia.org/wiki/File:Boxplot_vs_PDF.svg">Boxplot vs PDF</a>, <a href="https://creativecommons.org/licenses/by-sa/2.5/legalcode" rel="license">CC BY-SA 2.5</a></small></p>`,
        tags: []
    },
    {
        term: 'Learning Rate',
        link: `learning-rate`,
        content: /*html*/ `The learning rate describes how fast a system adapts to changes.`,
        tags: ['ml', 'dl']
    },
    {
        term: 'Instance-based Learning',
        link: `instance-based-learning`,
        content: /*html*/ `A system that basically 'learns by heart'. It predicts by looking at similar data. It requires a 'measure of similarity' function.`,
        tags: []
    },
    {
        term: 'Fitness / Utility function',
        link: `fitness-utility-function`,
        content: /*html*/ `A function that measures which values perform the best on a model.`,
        tags: []
    },
    {
        term: 'Precision',
        link: `precision`,
        content: /*html*/ `A measure of statistical variability. Is the fraction of relevant instances among the retrieved instances.
        $$ precision = {TP \\over TP + FP}$$`,
        tags: []
    },
    {
        term: 'Recall',
        link: `recall`,
        content: /*html*/ `Also: Sensitvity, True Positive Rate (TPR). Is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
        $$ recall = { TP \\over TP + FN}  $$`,
        tags: []
    },
    {
        term: 'F1 Score',
        link: `f1-score`,
        content: /*html*/ `Also: Harmonic mean.$$ F_1 = { TP \\over TP + { FN + FP \\over 2}}$$`,
        tags: []
    },
    {
        term: 'Linear Regression',
        link: `linear-regression`,
        content: /*html*/ `
        <p>Linear regression is a linear approach to find relations in data. Functions with more than two degrees of freedom are called <b>multiple linear regression</b>.</p>

        <p>
            Regular form 
        </p>
        $$ \\hat{y} = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + \\dots + \\theta_n x_n  $$
        Vectorized form
        $$ \\hat{y} = h_{\\theta}(\\textbf{x}) = \\theta^T \\cdot \\textbf{x} $$`,
        tags: []
    },
    {
        term: 'Mean Squared Error (MSE)',
        link: `mean-squared-error-mse`,
        content: /*html*/ `Cost function. Convex for a Linear Regression Models. 
        $$ MSE(\\textbf{X},h_{\\theta}) = \\frac{1}{m}{\\sum_{i=1}^{m}(\\hat{Y}_{i} - Y_i)^2}$$
        $$ J(\\theta) = \\frac{1}{m}{\\sum_{i=1}^{m}(h_{\\theta}(x^{(i)}) - y^{(i)})^2}$$
        `,
        tags: []
    },
    {
        term: 'Normal Equation',
        link: `normal-equation`,
        content: /*html*/ `<p>Closed form solution of the normal equation. It gives the result immediately, without any iteration. Due to the complexity of inverting a matrix $ O(n^{2.4}) $ training is slow, and all the data needs to fit into memory. However, prediction is just a dot product, and hence $ O(m) $.
        </p>
        <pre><code class="python">from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()</code></pre>
        $$ \\textbf{X} \\theta = \\textbf{y} $$  
        $$ \\textbf{X}^T X \\theta = \\textbf{X}^T \\cdot \\textbf{y} $$ 
        $$ \\hat{\\theta} = (\\textbf{X}^T \\cdot \\textbf{X})^{-1} \\cdot \\textbf{X}^T \\cdot \\textbf{y} $$`,
        tags: []
    },
    {
        term: 'Gradient Descent (GD)',
        link: `gradient-descent`,
        content: /*html*/ `<p>
            Gradient descent is a generic, first-order iterative optimization algorithm for finding optimal solutions. Goal is to iteratively modify parameters in order to minimize a cost function. To find a solution (a minimum), the algorithm takes steps into the negative direction of the derivative (down the slope), based of the current position. An important hyperparameter is the learning rate: the step size with which the algorithm proceeds into the negative direction of the derivative. <b>Tip:</b>Features should have a similar scale (StandardScalar) for better convergence properties.
        </p><img src="" alt="todo">`,
        tags: []
    },
    {
        term: 'Batch Gradient Descent',
        link: `batch-gradient-descent`,
        content: /*html*/ `
        <p>As the name suggest, with batch gradient descent we compute the gradient descent on a batch of data. We are computing the <i>partial derivatives</i> for all parameters of $ \\theta $. We basically evaluate how the cost function changes, if we slighlty tweak $ \\theta_j $. $j$ is a dimension within the hyperspace, in which we optimize. Adding up the results for every dimension gives us a weighted direction.</p>
        Partial derivative of the cost function
        $$ \\frac{\\delta}{\\delta\\theta_j}MSE(\\theta) = \\frac{2}{m}{\\sum_{i=1}^{m}(\\theta^T \\cdot \\textbf{x}^{(i)} - y^{(i)}) x_j^{(i)}}$$
        
        $$ \\frac{\\delta}{\\delta\\theta_j}MSE(\\theta) = \\frac{2}{m}{\\sum_{i=1}^{m}(h_{\\theta}(\\textbf{x}) - y^{(i)}) x_j^{(i)}}$$
        
        Gradient vector of the cost function
        $$ \\nabla_\\theta MSE(\\theta) = \\frac{2}{m}\\textbf{X}^{T} \\cdot (\\textbf{X} \\cdot \\theta - y)$$
        Gradient descent step
        $$ \\theta^{(\\text{next_step})} = \\theta - \\eta\\nabla_\\theta MSE(\\theta) $$`,
        tags: []
    },
    {
        term: 'Stochastic Gradient Descent (SGD)',
        link: `stoachastic-gradient-descent-sgd`,
        content: /*html*/ `<p>SGD picks random instances of the training data and computes the gradient on them. The algorithm will be more unregular then batch gradient descent, but eventually converge close to the same minimum, while being way more performant. The irregular jumps (due to the noise of single samples) can help escape local minimas, and the chance of finding the global minimum is higher than with BGD.</p>
        <pre><code class="python">from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=100, eta0=0.1)</code></pre>`,
        tags: []
    },
    {
        term: 'Simulated Annealing',
        link: ``,
        content: /*html*/ `The process of continously decreasing a learning rate, in order to allow an algorithm to settle at a minima.`,
        tags: []
    },
    {
        term: 'Mini-batch Gradient Descent',
        link: `mini-batch-gradient-descent-mbgd`,
        content: /*html*/ `Mini-bach GD conceptually does the same as BGD, just that it is not computed on the entire training set, but just a small random set.`,
        tags: []
    },
    {
        term: 'Polynomial Regression',
        link: `polynomial-regression`,
        content: /*html*/ `<p>Adding polynomial features of existing features, enables a linear regression model to fit polynomial data sets.</p>
        <pre><code class="python">from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
X_2nd_degree = poly_feature.fit_transform(X)</code></pre>`,
        tags: []
    },
    {
        term: 'Degrees of Freedom',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    },
    {
        term: '',
        link: ``,
        content: /*html*/ ``,
        tags: []
    }
];

var root = document.getElementById("root");

list.sort(compare)

list.forEach(item => {

    var a = document.createElement("a");
    a.setAttribute('href', '#' + item.link);
    a.name = item.link;

    var h = document.createElement("h2");
    h.classList.add("mt-10");
    h.innerHTML = item.term
    a.appendChild(h);

    var p = document.createElement("p");
    p.classList.add("lead");
    p.innerHTML = item.content

    root.appendChild(a);
    root.appendChild(p);
});



function compare(a, b) {
    const termA = a.term.toUpperCase();
    const termB = b.term.toUpperCase();

    let comparison = 0;
    if (termA > termB) {
        comparison = 1;
    } else if (termA < termB) {
        comparison = -1;
    }
    return comparison;
}