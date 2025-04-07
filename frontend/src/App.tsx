import { Provider } from 'jotai';
import { Toaster } from 'sonner';
import ImageGenerationPage from './components/ImageGeneration';



const App: React.FC = () => {
 
  
  return (
  //  <Provider>
  <>
    <ImageGenerationPage/>
    <Toaster position="top-right" />
    {/* </Provider> */}
    </>
  );
};

export default App;