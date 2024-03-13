import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


app.registerExtension({
    name: "Memeplex",

    async setup() {
        // queuePromptを呼び出すイベントリスナーを追加
        api.addEventListener("run_workflow", ({ detail }) => {
            //app.queuePrompt(0);
        });

        // TextInputノードのテキストを更新するイベントリスナーを追加
        api.addEventListener("update_text", ({ detail }) => {
            const nodes = app.graph.findNodesByType("GPT");
            for (const node of nodes) {
                console.log(detail)
                const text = detail["text"];
                console.log(node)
                node.widgets[3].value = text;
            }
        });
    }
})

app.registerExtension({
    name: "Memeplex",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TextInput") {
            //中略...
             nodeType.prototype.color = LGraphCanvas.node_colors.green.color;
             nodeType.prototype.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        }
         else if (nodeData.name === "GPT") {
             nodeType.prototype.color = LGraphCanvas.node_colors.green.color;
             nodeType.prototype.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
         }
    }
 })