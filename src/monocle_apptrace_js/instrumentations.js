const {
    InstrumentationBase,
    InstrumentationConfig,
    InstrumentationNodeModuleDefinition,
} = require('@opentelemetry/instrumentation')

const { trace, context, Tracer, SpanStatusCode } = require("@opentelemetry/api")

const { PACKAGE_LIST } = require("./packageList")

const { BatchSpanProcessor, ConsoleSpanExporter, NodeTracerProvider } = require("@opentelemetry/sdk-trace-node")

// import type * as AgentsModule from "langchain/agents";
// import type * as ToolsModule from "langchain/tools";

class MyInstrumentation extends InstrumentationBase {
    constructor(config = {}) {
        super('MyInstrumentation', "1.0", config);
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
        PACKAGE_LIST.forEach(element => {
            const module = new InstrumentationNodeModuleDefinition(
                element.packagePath,
                ['*'],
                this._getOnPatchMain(element.className, element.methodName, element.attributeSetter).bind(this),
                this._onUnPatchMain.bind(this),
            );
            modules.push(module)
        });

        // in case you need to patch additional files - this is optional
        //module.files.push(this._addPatchingMethod());

        return modules;
    }

    manuallyStartInstrument() {
        this._onPatchMain(OpenAIModule)
    }

    _getOnPatchMain(exportedClassName, methodName, attributeSetter) {

        return function (moduleExports) {
            // console.log("exportedClassName:" + exportedClassName + ", methodName:" + methodName)
            this._wrap(
                moduleExports[exportedClassName].prototype,
                methodName,
                this._patchMainMethodName(attributeSetter)
            );
            return moduleExports;
        }
    }

    _onPatchMain(moduleExports) {
        console.log("init2")
        this._wrap(
            moduleExports["BaseChatModel"].prototype,
            'invoke',
            this._patchMainMethodName()
        );
        return moduleExports;
    }

    _onUnPatchMain(moduleExports) {
        // this._unwrap(moduleExports.BaseChatModel.prototype, 'invoke');
    }

    _patchMainMethodName = (attributeSetter) => {
        const tracer = this.tracer
        const context_global = context
        console.log('mainMethodName1');
        // const plugin = this;
        return function mainMethodName(original) {
            return function patchMainMethodName() {
                console.log('mainMethodName2', arguments);
                return tracer.startActiveSpan(
                    ``,
                    async (span) => {
                        // span.setAttribute(, spanKind);

                        // const execContext = trace.setSpan(context.active(), span);

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


exports.MyInstrumentation = MyInstrumentation

exports.setupMonocle = setupMonocle = () => {
    // Later, but before the module to instrument is required
    const exporter = new ConsoleSpanExporter();
    const monocleProcessor = new BatchSpanProcessor(
        new ConsoleSpanExporter(),
        config = {
            scheduledDelayMillis: 20
        });
    const traceProvider = new NodeTracerProvider()
    traceProvider.addSpanProcessor(monocleProcessor)
    const myInstrumentation = new MyInstrumentation();

    myInstrumentation.setTracerProvider(traceProvider); // this is optional, only if global TracerProvider shouldn't be used
    // myInstrumentation.setMeterProvider(meterProvider); // this is optional
    myInstrumentation.enable();
    // or use Auto Loader

    // return exporter
    return exporter;
}
