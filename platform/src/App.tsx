// src/App.tsx

import { Route, Router } from "@solidjs/router";
import Index from "./pages/Index.tsx";
import "./App.css";

const App = () => {
  return (
    <Router>
      <Route path="/" component={Index} />
      {/* <Route path="/:selectedDinosaur" component={Dinosaur} /> */}
    </Router>
  );
};

export default App;
