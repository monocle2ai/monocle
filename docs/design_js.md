## How commonsJS modules are imported in javascript

A sample javascript class looks as following

```javascript
const sampleModule = require("sample-module")

sampleModule.doSomething()
```

All the exported functions, classes, variables from sample-module are members of the `exports` object.

```javascript
exports.doSomething = function()
{
    // do work
}
```

## How we intercept the exported members from sample-module

This `require` function can be overwritten as following

```javascript
var Module = require('module');
var originalRequire = Module.prototype.require;

Module.prototype.require = function(){
  //do your instrumentation here
  return originalRequire.apply(this, arguments);
};
```

By hooking into the require function we are able to override the exported members from sample-module.
This enables us to instrument the members.

## Example

```javascript
var Module = require('module');
// cache the original require
var originalRequire = Module.prototype.require;

const instrumentationWrapper = function(functionToBeWrapped) {
    return function() {
        // start timer to find the time taken by function call
        startTimer()
        const returned_value = functionToBeWrapped.apply(this, arguments)
        endTimer()
        return returned_value;
    }
}

Module.prototype.require = function() {
    const returnedValue = originalRequire.apply(this, arguments);

    // check if this is the module that you want to instrument
    if (arguments[0] == "sample-module") {
        // overwrite the function with your instrumentation wrapper
        returnedValue.doSomething = instrumentationWrapper(returnedValue.doSomething)
    }
    return returnedValue;
};
```








