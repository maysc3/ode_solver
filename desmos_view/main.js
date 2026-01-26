const elt = document.getElementById('calculator');
const calculator = Desmos.GraphingCalculator(elt, {
  expressions: true,
  settingsMenu: false,
  zoomButtons: true
});

fetch('data/solution_data.json')
  .then(response => response.json())
  .then(data => {
    calculator.setExpression({
      id: 'solution-table',
      type: 'table',
      columns: [
        {
          latex: 't',
          values: data.t.map(x => Number(x.toFixed(4)))
        },
        {
          latex: 'y',
          values: data.y.map(x => Number(x.toFixed(4)))
        }
      ]
    });
  })
  .catch(err => console.error("Failed to load data", err));