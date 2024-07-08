const {
    InstrumentationBase,
    InstrumentationNodeModuleDefinition,
} = require('@opentelemetry/instrumentation')
const { context, } = require("@opentelemetry/api")
const { Resource } = require("@opentelemetry/resources")
const { NodeTracerProvider } = require("@opentelemetry/sdk-trace-node")
const { AsyncHooksContextManager } = require("@opentelemetry/context-async-hooks")
const { combinedPackages } = require("./common/packages")


class MonocleInstrumentation extends InstrumentationBase {
    constructor(config = {}) {
        super('MonocleInstrumentation', "1.0", config);
    }
    modules = []

    /**
     * Init method will be called when the plugin is constructed.
     * It returns an `InstrumentationNodeModuleDefinition` which describes
     *   the node module to be instrumented and patched.
     * It may also return a list of `InstrumentationNodeModuleDefinition`s if
     *   the plugin should patch multiple modules or versions.
     */
    init() {
        const modules = []
        combinedPackages.forEach(packages => {
            packages.forEach(element => {
                const module = new InstrumentationNodeModuleDefinition(
                    element.packagePath,
                    ['*'],
                    this._getOnPatchMain(element.className, element.methodName, element.attributeSetter).bind(this),
                );
                modules.push(module)
            });
        })

        return modules;
    }

    _getOnPatchMain(exportedClassName, methodName, attributeSetter) {

        return function (moduleExports) {
            this._wrap(
                moduleExports[exportedClassName].prototype,
                methodName,
                this._patchMainMethodName(attributeSetter)
            );
            return moduleExports;
        }
    }

    _patchMainMethodName = (attributeSetter) => {
        const tracer = this.tracer
        return function mainMethodName(original) {
            return function patchMainMethodName() {
                return tracer.startActiveSpan(
                    "",
                    async (span) => {

                        const ret_val = await original.apply(this, arguments);

                        if (attributeSetter) {
                            attributeSetter(
                                {
                                    returnedValue: ret_val,
                                    arguments: arguments,
                                    classInstance: this,
                                    span: span
                                })
                        }

                        span.end()

                        return ret_val
                    }
                );

            };
        };
    }
}

exports.setupOkahu = setupOkahu = (
    workflowName,
    spanProcessors = [],
    wrapperMethods = []
) => {
    const resource = new Resource(attributes = {
        SERVICE_NAME: workflowName
    })
    const contextManager = new AsyncHooksContextManager();
    contextManager.enable();
    context.setGlobalContextManager(contextManager);
    const traceProvider = new NodeTracerProvider({
        resource: resource
    })
    for (let processor of spanProcessors)
        traceProvider.addSpanProcessor(processor)

    const myInstrumentation = new MonocleInstrumentation();

    myInstrumentation.setTracerProvider(traceProvider);
    myInstrumentation.enable();

}
